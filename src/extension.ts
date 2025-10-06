import * as vscode from 'vscode';
import { GrepalClient } from './grepalClient';
let grepalClient: GrepalClient;
let statusBarItem: vscode.StatusBarItem;
let debounceTimer: NodeJS.Timeout | undefined;
let lastAnalyzedCode: string = '';
let popupQueue: Array<{issue: any, index: number}> = [];
let isShowingPopup = false;
let currentAnalysisPopup: Thenable<string | undefined> | undefined;
let analysisHistory: Array<{timestamp: Date, filename: string, language: string, snarkComment: string, issues: any[], overallScore: number}> = [];
let historyPanel: vscode.WebviewPanel | undefined;

// Analyzed lines buffer to prevent re-analyzing fixed code sections
let analyzedLinesBuffer: Map<string, Array<{startLine: number, endLine: number, timestamp: Date}>> = new Map();

function hashCode(str: string): number {
    let hash = 0;
    if (str.length === 0) return hash;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('Grepal is now active!');
    
    grepalClient = new GrepalClient();
    
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(bug) Grepal: Ready";
    statusBarItem.tooltip = "Grepal - Click to view analysis history";
    statusBarItem.command = 'grepal.showInsights';
    statusBarItem.show();
    
    const enableCommand = vscode.commands.registerCommand('grepal.enable', async () => {
        vscode.workspace.getConfiguration('grepal').update('enabled', true, true);
        statusBarItem.text = "$(bug) Grepal: Active";
        
        try {
            await grepalClient.start();
            vscode.window.showInformationMessage('Grepal enabled! Ready to roast your bugs.');
        } catch (error) {
            vscode.window.showWarningMessage('Grepal enabled but server connection failed. Make sure server is running.');
        }
    });
    
    const disableCommand = vscode.commands.registerCommand('grepal.disable', () => {
        vscode.workspace.getConfiguration('grepal').update('enabled', false, true);
        vscode.window.showInformationMessage('Grepal disabled. Your bugs are safe... for now.');
        statusBarItem.text = "$(bug) Grepal: Disabled";
        grepalClient.stop();
    });
    
    const showInsightsCommand = vscode.commands.registerCommand('grepal.showInsights', () => {
        if (historyPanel) {
            historyPanel.reveal(vscode.ViewColumn.Two);
        } else {
            historyPanel = vscode.window.createWebviewPanel(
                'grepalHistory',
                'Grepal Analysis History',
                vscode.ViewColumn.Two,
                { enableScripts: true }
            );
            
            historyPanel.webview.html = getHistoryWebviewContent();
            
            historyPanel.onDidDispose(() => {
                historyPanel = undefined;
            });
            
            historyPanel.webview.onDidReceiveMessage(
                message => {
                    switch (message.command) {
                        case 'analyze':
                            analyzeCurrentFile();
                            break;
                        case 'clearHistory':
                            clearHistory();
                            break;
                        case 'clearBuffer':
                            clearAnalyzedLinesBuffer();
                            break;
                    }
                },
                undefined,
                context.subscriptions
            );
        }
    });
    
    const textChangeListener = vscode.workspace.onDidChangeTextDocument(async (event) => {
        const config = vscode.workspace.getConfiguration('grepal');
        if (!config.get('enabled')) return;
        
        if (event.document === vscode.window.activeTextEditor?.document) {
            if (debounceTimer) {
                clearTimeout(debounceTimer);
            }
            
            debounceTimer = setTimeout(() => {
                const currentCode = event.document.getText();
                const currentHash = hashCode(currentCode);
                const lastHash = hashCode(lastAnalyzedCode || '');
                
                if (currentHash !== lastHash) {
                    console.log('Grepal: Debounced analysis triggered');
                    analyzeCurrentFile();
                }
            }, 1500); // Reduced from 2000ms to 1500ms for faster response
        }
    });
    
    context.subscriptions.push(
        enableCommand,
        disableCommand, 
        showInsightsCommand,
        textChangeListener,
        statusBarItem
    );
    
    const config = vscode.workspace.getConfiguration('grepal');
    if (config.get('enabled')) {
        grepalClient.start().catch(() => {
            console.log('Failed to connect to server on startup, will retry on first analysis');
        });
        statusBarItem.text = "$(bug) Grepal: Active";
    }
}

function showComprehensivePopup(issues: any[], snarkComment: string): Thenable<string | undefined> {
    // Dispose any existing popup first to ensure only one at a time
    if (currentAnalysisPopup) {
        console.log('Grepal: Dismissing existing popup to show new analysis');
    }
    
    const severityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
    const sortedIssues = issues.sort((a, b) => (severityOrder[b.severity as keyof typeof severityOrder] || 1) - (severityOrder[a.severity as keyof typeof severityOrder] || 1));
    
    const issueList = sortedIssues.slice(0, 5).map((issue, index) => {
        const icon = issue.severity === 'critical' ? 'CRITICAL' : 
                   issue.severity === 'high' ? 'HIGH' : 
                   issue.severity === 'medium' ? 'MEDIUM' : 'LOW';
        return `${icon} Line ${issue.line}: ${issue.message.substring(0, 80)}...`;
    }).join('\n');
    
    const moreIssues = issues.length > 5 ? `\n\n...and ${issues.length - 5} more issues!` : '';
    
    currentAnalysisPopup = vscode.window.showWarningMessage(
        `Grepal found ${issues.length} issues:\n\n${issueList}${moreIssues}\n\n"${snarkComment}"`,
        'Show Individual Issues', 'Fix All', 'Clear Buffer & Re-analyze'
    ).then(selection => {
        currentAnalysisPopup = undefined;
        
        if (selection === 'Show Individual Issues') {
            // Start showing individual popups sequentially
            if (!isShowingPopup && popupQueue.length > 0) {
                console.log('Grepal: Starting individual issue popup sequence');
                showNextPopup(snarkComment);
            }
        } else if (selection === 'Fix All') {
            vscode.window.showInformationMessage(
                'Grepal: "If only there was a magic button... Start with the critical/high severity ones!"'
            );
        } else if (selection === 'Clear Buffer & Re-analyze') {
            clearAnalyzedLinesBuffer();
            setTimeout(() => analyzeCurrentFile(), 500);
        }
        
        return selection;
    });
    
    return currentAnalysisPopup;
}

function showNextPopup(finalSnarkComment?: string) {
    if (popupQueue.length === 0) {
        isShowingPopup = false;
        
        if (finalSnarkComment && finalSnarkComment.trim()) {
            setTimeout(() => {
                vscode.window.showInformationMessage(
                    `Grepal's final verdict: "${finalSnarkComment}"`
                );
            }, 500);
        }
        return;
    }
    
    isShowingPopup = true;
    const { issue } = popupQueue.shift()!;
    
    const severityIcon = issue.severity === 'critical' ? 'CRITICAL' : 
                       issue.severity === 'high' ? 'HIGH' : 
                       issue.severity === 'medium' ? 'MEDIUM' : 'LOW';
    
    const meanerMessage = makeMessageMeaner(issue.message);
    const remainingIssues = popupQueue.length > 0 ? ` (${popupQueue.length + 1} issues remaining)` : '';
    
    console.log(`Grepal: Showing popup ${popupQueue.length + 1} - ${severityIcon}`);
    
    setTimeout(() => {
        vscode.window.showWarningMessage(
            `${severityIcon} Line ${issue.line}: ${meanerMessage}${remainingIssues}`,
            'Show Fix (I need help)', 'Next Issue'
        ).then(selection => {
            if (selection === 'Show Fix (I need help)') {
                const meanerFix = makeFixMeaner(issue.suggestion);
                vscode.window.showInformationMessage(
                    `${meanerFix}`
                );
                setTimeout(() => showNextPopup(finalSnarkComment), 1500);
            } else {
                // Default behavior: proceed to next issue
                setTimeout(() => showNextPopup(finalSnarkComment), 800);
            }
        });
    }, 200);
}

function makeMessageMeaner(message: string): string {
    const meanerPrefixes = [
        "Seriously?",
        "Are you kidding me?", 
        "What were you thinking?",
        "This is embarrassing.",
        "Oh come on!",
        "Really? REALLY?",
        "Did you even try?",
        "This hurts to look at.",
        "My eyes are bleeding.",
        "This is painful."
    ];
    
    const meanerSuffixes = [
        "What a rookie mistake!",
        "Even a beginner wouldn't do this.",
        "This is Computer Science 101.",
        "Did you skip coding bootcamp?",
        "Time to go back to tutorials.",
        "Maybe try a different career?",
        "Your rubber duck is disappointed.",
        "Stack Overflow won't even help with this.",
        "This makes me question humanity.",
        "I've lost faith in developers."
    ];
    
    const prefix = meanerPrefixes[Math.floor(Math.random() * meanerPrefixes.length)];
    const suffix = meanerSuffixes[Math.floor(Math.random() * meanerSuffixes.length)];
    
    return `${prefix} ${message} ${suffix}`;
}

function makeFixMeaner(suggestion: string): string {
    const meanerFixPrefixes = [
        "Since you clearly need help:",
        "Let me spell it out for you:",
        "Here's what you should have done:",
        "Pay attention this time:",
        "Try this (if you can handle it):",
        "Fine, I'll do your job:",
        "Here's the obvious fix:",
        "Even my grandma knows this:",
        "This should be common sense:",
        "Sigh... here's how you fix it:"
    ];
    
    const prefix = meanerFixPrefixes[Math.floor(Math.random() * meanerFixPrefixes.length)];
    return `${prefix} ${suggestion} (You're welcome.)`;
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    const document = editor.document;
    const code = document.getText();
    
    // Skip empty or very short files
    if (code.trim().length < 10) {
        return;
    }
    
    // For debugging: always analyze if code changed significantly
    const codeHash = hashCode(code);
    const lastHash = hashCode(lastAnalyzedCode || '');
    
    if (codeHash === lastHash) {
        console.log('Grepal: Code unchanged, skipping analysis');
        return;
    }
    
    // Filter out already analyzed lines
    const filteredCode = filterAnalyzedLines(document.fileName, code);
    if (filteredCode.trim().length < 10) {
        console.log('Grepal: Most code already analyzed, skipping');
        statusBarItem.text = "$(check) Grepal: Already analyzed";
        return;
    }
    
    console.log(`Grepal: Analyzing ${document.languageId} code (${code.length} chars, ${filteredCode.length} new chars)`);
    lastAnalyzedCode = code;
    
    try {
        statusBarItem.text = "$(sync~spin) Grepal: Analyzing...";
        
        // Use filtered code for analysis but still pass full context
        const analysis = await grepalClient.analyzeCode({
            code: filteredCode,
            language: document.languageId,
            filename: document.fileName
        });
        
        console.log(`Grepal: Analysis complete - found ${analysis.issues.length} issues`);
        
        // Add analyzed regions to buffer based on found issues
        if (analysis.issues.length > 0) {
            addAnalyzedLinesFromIssues(document.fileName, analysis.issues, document.lineCount);
        } else {
            // If no issues found, add the entire analyzed region to buffer
            addAnalyzedLinesRegion(document.fileName, 1, document.lineCount);
        }
        
        // Add to history regardless of whether issues were found
        addToHistory({
            timestamp: new Date(),
            filename: document.fileName.split('/').pop() || 'Unknown File',
            language: document.languageId,
            snarkComment: analysis.snarkComment,
            issues: analysis.issues,
            overallScore: analysis.overallScore
        });
        
        if (analysis.issues.length > 0) {
            statusBarItem.text = `$(warning) Grepal: ${analysis.issues.length} issues`;
            console.log('Grepal: Issues found:', analysis.issues.map(i => `Line ${i.line}: ${i.message}`));
            
            popupQueue = analysis.issues.map((issue, index) => ({ issue, index }));
            
            // Only show one popup at a time - the comprehensive summary
            console.log('Grepal: Showing single comprehensive popup');
            showComprehensivePopup(analysis.issues, analysis.snarkComment);
            
        } else {
            statusBarItem.text = "$(check) Grepal: Looking good!";
            console.log('Grepal: No issues found - code looks clean!');
            
            const encouragingMessages = [
                "Nice work! No issues found.",
                "Clean code! Grepal approves.",
                "Looking good! Keep it up.",
                "Solid code! No complaints here.",
                "Perfect! No bugs detected."
            ];
            
            const randomMessage = encouragingMessages[Math.floor(Math.random() * encouragingMessages.length)];
            
            if (Math.random() < 0.3) {
                vscode.window.showInformationMessage(randomMessage);
            }
        }
        
    } catch (error: any) {
        console.error('Grepal analysis failed:', error);
        statusBarItem.text = "$(error) Grepal: Error";
        
        // More detailed error message
        const errorMessage = error?.message || 'Unknown error';
        console.error('Grepal error details:', errorMessage);
        
        if (errorMessage.includes('ECONNREFUSED') || errorMessage.includes('fetch')) {
            vscode.window.showErrorMessage('Grepal server is not responding. Make sure server is running on http://localhost:8000');
        } else {
            vscode.window.showErrorMessage(`Grepal analysis failed: ${errorMessage}`);
        }
    }
}

function addToHistory(entry: {timestamp: Date, filename: string, language: string, snarkComment: string, issues: any[], overallScore: number}) {
    analysisHistory.unshift(entry); // Add to beginning of array (most recent first)
    
    // Keep only last 50 entries to prevent memory issues
    if (analysisHistory.length > 50) {
        analysisHistory = analysisHistory.slice(0, 50);
    }
    
    // Update webview if it's open
    if (historyPanel) {
        historyPanel.webview.html = getHistoryWebviewContent();
    }
}

function clearHistory() {
    analysisHistory = [];
    if (historyPanel) {
        historyPanel.webview.html = getHistoryWebviewContent();
    }
}

function getHistoryWebviewContent(): string {
    const historyHtml = analysisHistory.length === 0 
        ? '<div class="empty-state">No analysis history yet. Run Grepal on some code to see results here!</div>'
        : analysisHistory.map((entry, index) => {
            const timeAgo = formatTimeAgo(entry.timestamp);
            const scoreClass = getScoreClass(entry.overallScore);
            const issueCount = entry.issues.length;
            
            const issuesHtml = entry.issues.length > 0 
                ? entry.issues.map(issue => `
                    <div class="issue-item">
                        <div class="issue-header">
                            <span class="severity-${issue.severity}">${issue.severity.toUpperCase()}</span>
                            <span class="issue-line">Line ${issue.line}</span>
                        </div>
                        <div class="issue-message">${escapeHtml(issue.message)}</div>
                        <div class="issue-suggestion">
                            <strong>Fix:</strong> ${escapeHtml(issue.suggestion)}
                        </div>
                    </div>
                `).join('')
                : '<div class="no-issues">✨ No issues found - clean code!</div>';
            
            return `
                <div class="history-entry">
                    <div class="entry-header">
                        <div class="entry-info">
                            <span class="filename">${escapeHtml(entry.filename)}</span>
                            <span class="language">${entry.language}</span>
                            <span class="timestamp">${timeAgo}</span>
                        </div>
                        <div class="score ${scoreClass}">${entry.overallScore}/100</div>
                    </div>
                    
                    <div class="snark-comment">${escapeHtml(entry.snarkComment)}</div>
                    
                    <div class="issues-section">
                        <div class="issues-header">Issues Found: ${issueCount}</div>
                        ${issuesHtml}
                    </div>
                </div>
            `;
        }).join('');
    
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grepal Analysis History</title>
        <style>
            body {
                font-family: var(--vscode-font-family);
                color: var(--vscode-foreground);
                background: var(--vscode-editor-background);
                padding: 20px;
                line-height: 1.4;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid var(--vscode-panel-border);
            }
            
            .title {
                font-size: 24px;
                font-weight: bold;
                color: var(--vscode-foreground);
            }
            
            .actions {
                display: flex;
                gap: 10px;
            }
            
            .btn {
                background: var(--vscode-button-background);
                color: var(--vscode-button-foreground);
                border: none;
                padding: 8px 16px;
                cursor: pointer;
                border-radius: 3px;
                font-size: 13px;
            }
            
            .btn:hover {
                background: var(--vscode-button-hoverBackground);
            }
            
            .btn-secondary {
                background: var(--vscode-button-secondaryBackground);
                color: var(--vscode-button-secondaryForeground);
            }
            
            .btn-secondary:hover {
                background: var(--vscode-button-secondaryHoverBackground);
            }
            
            .empty-state {
                text-align: center;
                padding: 40px 20px;
                color: var(--vscode-descriptionForeground);
                font-style: italic;
            }
            
            .history-entry {
                background: var(--vscode-editor-inactiveSelectionBackground);
                border: 1px solid var(--vscode-panel-border);
                border-radius: 6px;
                margin-bottom: 16px;
                padding: 16px;
            }
            
            .entry-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }
            
            .entry-info {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .filename {
                font-weight: bold;
                color: var(--vscode-foreground);
            }
            
            .language {
                background: var(--vscode-badge-background);
                color: var(--vscode-badge-foreground);
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
                text-transform: uppercase;
            }
            
            .timestamp {
                color: var(--vscode-descriptionForeground);
                font-size: 12px;
            }
            
            .score {
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            
            .score.excellent { background: #1a7f1a; color: white; }
            .score.good { background: #007acc; color: white; }
            .score.fair { background: #ff8c00; color: white; }
            .score.poor { background: #d73a49; color: white; }
            
            .snark-comment {
                font-style: italic;
                color: var(--vscode-foreground);
                background: var(--vscode-textBlockQuote-background);
                border-left: 4px solid var(--vscode-textBlockQuote-border);
                padding: 12px;
                margin: 12px 0;
                border-radius: 0 4px 4px 0;
            }
            
            .issues-section {
                margin-top: 16px;
            }
            
            .issues-header {
                font-weight: bold;
                margin-bottom: 8px;
                color: var(--vscode-foreground);
            }
            
            .no-issues {
                color: #1a7f1a;
                font-weight: bold;
                text-align: center;
                padding: 12px;
            }
            
            .issue-item {
                background: var(--vscode-editor-background);
                border: 1px solid var(--vscode-panel-border);
                border-radius: 4px;
                padding: 12px;
                margin-bottom: 8px;
            }
            
            .issue-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            
            .severity-critical { color: #d73a49; font-weight: bold; }
            .severity-high { color: #ff8c00; font-weight: bold; }
            .severity-medium { color: #007acc; font-weight: bold; }
            .severity-low { color: var(--vscode-descriptionForeground); }
            
            .issue-line {
                color: var(--vscode-descriptionForeground);
                font-size: 12px;
            }
            
            .issue-message {
                color: var(--vscode-foreground);
                margin-bottom: 8px;
            }
            
            .issue-suggestion {
                color: var(--vscode-descriptionForeground);
                font-size: 13px;
                background: var(--vscode-textCodeBlock-background);
                padding: 8px;
                border-radius: 3px;
            }
            
            .stats {
                background: var(--vscode-editor-inactiveSelectionBackground);
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 20px;
                font-size: 13px;
                color: var(--vscode-descriptionForeground);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">🐛 Grepal Analysis History</div>
            <div class="actions">
                <button class="btn" onclick="analyzeCode()">Analyze Current File</button>
                <button class="btn btn-secondary" onclick="clearHistory()">Clear History</button>
                <button class="btn btn-secondary" onclick="clearBuffer()">Clear Buffer</button>
            </div>
        </div>
        
        <div class="stats">
            Total analyses: ${analysisHistory.length} | Average score: ${getAverageScore()}/100 | Buffer: ${getAnalyzedLinesStats()}
        </div>
        
        <div class="history-container">
            ${historyHtml}
        </div>
        
        <script>
            const vscode = acquireVsCodeApi();
            
            function analyzeCode() {
                vscode.postMessage({ command: 'analyze' });
            }
            
            function clearHistory() {
                if (confirm('Clear all analysis history?')) {
                    vscode.postMessage({ command: 'clearHistory' });
                }
            }
            
            function clearBuffer() {
                if (confirm('Clear analyzed lines buffer? This will cause Grepal to re-analyze all code sections.')) {
                    vscode.postMessage({ command: 'clearBuffer' });
                }
            }
        </script>
    </body>
    </html>
    `;
}

function formatTimeAgo(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
}

function getScoreClass(score: number): string {
    if (score >= 90) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 50) return 'fair';
    return 'poor';
}

function getAverageScore(): string {
    if (analysisHistory.length === 0) return '0';
    const avg = analysisHistory.reduce((sum, entry) => sum + entry.overallScore, 0) / analysisHistory.length;
    return avg.toFixed(1);
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Analyzed Lines Buffer Management Functions

function filterAnalyzedLines(fileName: string, code: string): string {
    const analyzedRanges = analyzedLinesBuffer.get(fileName) || [];
    if (analyzedRanges.length === 0) {
        return code; // No previously analyzed lines
    }
    
    const lines = code.split('\n');
    const filteredLines: string[] = [];
    
    for (let i = 0; i < lines.length; i++) {
        const lineNumber = i + 1; // Line numbers start at 1
        const isInAnalyzedRange = analyzedRanges.some(range => 
            lineNumber >= range.startLine && lineNumber <= range.endLine
        );
        
        if (!isInAnalyzedRange) {
            filteredLines.push(lines[i]);
        }
    }
    
    console.log(`Grepal: Filtered ${fileName} - ${lines.length} total lines, ${filteredLines.length} new lines to analyze`);
    return filteredLines.join('\n');
}

function addAnalyzedLinesFromIssues(fileName: string, issues: any[], totalLines: number) {
    if (!analyzedLinesBuffer.has(fileName)) {
        analyzedLinesBuffer.set(fileName, []);
    }
    
    const ranges = analyzedLinesBuffer.get(fileName)!;
    const now = new Date();
    
    // Add regions around each issue (with some buffer)
    issues.forEach(issue => {
        const issueLineNumber = parseInt(issue.line) || 1;
        const bufferSize = 10; // Lines of context around each issue
        const startLine = Math.max(1, issueLineNumber - bufferSize);
        const endLine = Math.min(totalLines, issueLineNumber + bufferSize);
        
        // Check if this range overlaps with existing ranges
        const overlappingRange = ranges.find(range => 
            (startLine <= range.endLine && endLine >= range.startLine)
        );
        
        if (overlappingRange) {
            // Extend existing range
            overlappingRange.startLine = Math.min(overlappingRange.startLine, startLine);
            overlappingRange.endLine = Math.max(overlappingRange.endLine, endLine);
            overlappingRange.timestamp = now;
        } else {
            // Add new range
            ranges.push({ startLine, endLine, timestamp: now });
        }
        
        console.log(`Grepal: Added analyzed range ${startLine}-${endLine} for issue at line ${issueLineNumber}`);
    });
    
    // Clean up old ranges (older than 1 hour)
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    const filteredRanges = ranges.filter(range => range.timestamp > oneHourAgo);
    analyzedLinesBuffer.set(fileName, filteredRanges);
}

function addAnalyzedLinesRegion(fileName: string, startLine: number, endLine: number) {
    if (!analyzedLinesBuffer.has(fileName)) {
        analyzedLinesBuffer.set(fileName, []);
    }
    
    const ranges = analyzedLinesBuffer.get(fileName)!;
    const now = new Date();
    
    // Check for overlapping ranges
    const overlappingRange = ranges.find(range => 
        (startLine <= range.endLine && endLine >= range.startLine)
    );
    
    if (overlappingRange) {
        // Extend existing range
        overlappingRange.startLine = Math.min(overlappingRange.startLine, startLine);
        overlappingRange.endLine = Math.max(overlappingRange.endLine, endLine);
        overlappingRange.timestamp = now;
    } else {
        // Add new range
        ranges.push({ startLine, endLine, timestamp: now });
    }
    
    console.log(`Grepal: Added analyzed region ${startLine}-${endLine} for ${fileName}`);
}

function clearAnalyzedLinesBuffer(fileName?: string) {
    if (fileName) {
        analyzedLinesBuffer.delete(fileName);
        console.log(`Grepal: Cleared analyzed lines buffer for ${fileName}`);
    } else {
        analyzedLinesBuffer.clear();
        console.log('Grepal: Cleared all analyzed lines buffers');
        vscode.window.showInformationMessage('Grepal: Analysis buffer cleared. Will re-analyze all code.');
    }
}

function getAnalyzedLinesStats(): string {
    const totalFiles = analyzedLinesBuffer.size;
    let totalRanges = 0;
    let totalLines = 0;
    
    analyzedLinesBuffer.forEach((ranges, fileName) => {
        totalRanges += ranges.length;
        ranges.forEach(range => {
            totalLines += (range.endLine - range.startLine + 1);
        });
    });
    
    return `Files: ${totalFiles}, Ranges: ${totalRanges}, Lines: ${totalLines}`;
}

export function deactivate() {
    if (grepalClient) {
        grepalClient.stop();
    }
    if (historyPanel) {
        historyPanel.dispose();
    }
    console.log('Grepal deactivated');
}
