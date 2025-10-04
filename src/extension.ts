import * as vscode from 'vscode';
import { GrepalClient } from './grepalClient';
let grepalClient: GrepalClient;
let statusBarItem: vscode.StatusBarItem;
let debounceTimer: NodeJS.Timeout | undefined;
let lastAnalyzedCode: string = '';
let popupQueue: Array<{issue: any, index: number}> = [];
let isShowingPopup = false;

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
    statusBarItem.tooltip = "Grepal - Click to configure";
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
        const panel = vscode.window.createWebviewPanel(
            'grepalInsights',
            'Grepal Debug Insights',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );
        
        panel.webview.html = getWebviewContent();
        
        panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'analyze':
                        analyzeCurrentFile();
                        break;
                }
            },
            undefined,
            context.subscriptions
        );
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

function showComprehensivePopup(issues: any[], snarkComment: string) {
    const severityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
    const sortedIssues = issues.sort((a, b) => (severityOrder[b.severity as keyof typeof severityOrder] || 1) - (severityOrder[a.severity as keyof typeof severityOrder] || 1));
    
    const issueList = sortedIssues.slice(0, 5).map((issue, index) => {
        const icon = issue.severity === 'critical' ? 'CRITICAL' : 
                   issue.severity === 'high' ? 'HIGH' : 
                   issue.severity === 'medium' ? 'MEDIUM' : 'LOW';
        return `${icon} Line ${issue.line}: ${issue.message.substring(0, 80)}...`;
    }).join('\n');
    
    const moreIssues = issues.length > 5 ? `\n\n...and ${issues.length - 5} more issues!` : '';
    
    vscode.window.showWarningMessage(
        `Grepal found ${issues.length} issues:\n\n${issueList}${moreIssues}\n\n"${snarkComment}"`,
        'Show Individual Issues', 'Fix All', 'I Give Up'
    ).then(selection => {
        if (selection === 'Fix All') {
            vscode.window.showInformationMessage(
                'Grepal: "If only there was a magic button... Start with the critical/high severity ones!"'
            );
        } else if (selection === 'I Give Up') {
            vscode.window.showInformationMessage(
                'Grepal: "Wise choice. Your code was hopeless anyway."'
            );
        }
    });
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
            'Show Fix (I need help)', 'Next Roast', 'Stop the Pain'
        ).then(selection => {
            if (selection === 'Show Fix (I need help)') {
                const meanerFix = makeFixMeaner(issue.suggestion);
                vscode.window.showInformationMessage(
                    `${meanerFix}`
                );
                setTimeout(() => showNextPopup(finalSnarkComment), 1500);
            } else if (selection === 'Stop the Pain') {
                popupQueue = [];
                isShowingPopup = false;
                vscode.window.showInformationMessage(
                    `Grepal: "Can't handle the truth? Your code is still broken, but I'll spare you... for now."`
                );
            } else {
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
    
    console.log(`Grepal: Analyzing ${document.languageId} code (${code.length} chars)`);
    lastAnalyzedCode = code;
    
    try {
        statusBarItem.text = "$(sync~spin) Grepal: Analyzing...";
        
        const analysis = await grepalClient.analyzeCode({
            code,
            language: document.languageId,
            filename: document.fileName
        });
        
        console.log(`Grepal: Analysis complete - found ${analysis.issues.length} issues`);
        
        if (analysis.issues.length > 0) {
            statusBarItem.text = `$(warning) Grepal: ${analysis.issues.length} issues`;
            console.log('Grepal: Issues found:', analysis.issues.map(i => `Line ${i.line}: ${i.message}`));
            
            popupQueue = analysis.issues.map((issue, index) => ({ issue, index }));
            
            if (!isShowingPopup) {
                console.log('Grepal: Starting popup display');
                
                // Show comprehensive summary first, then individual popups
                showComprehensivePopup(analysis.issues, analysis.snarkComment);
                
                // Then show individual popups after a delay
                setTimeout(() => {
                    showNextPopup(analysis.snarkComment);
                }, 2000);
            }
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

function getWebviewContent(): string {
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grepal Insights</title>
        <style>
            body {
                font-family: var(--vscode-font-family);
                color: var(--vscode-foreground);
                background: var(--vscode-editor-background);
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .logo {
                font-size: 48px;
                margin-bottom: 10px;
            }
            .tagline {
                font-style: italic;
                color: var(--vscode-descriptionForeground);
            }
            .analyze-btn {
                background: var(--vscode-button-background);
                color: var(--vscode-button-foreground);
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                border-radius: 3px;
                font-size: 16px;
            }
            .analyze-btn:hover {
                background: var(--vscode-button-hoverBackground);
            }
            .insights {
                margin-top: 20px;
                padding: 15px;
                background: var(--vscode-editor-inactiveSelectionBackground);
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">Grepal</div>
            <h1>Grepal Debug Buddy</h1>
            <p class="tagline">"Your code is bad, and you should feel bad... but here's how to fix it."</p>
        </div>
        
        <div style="text-align: center;">
            <button class="analyze-btn" onclick="analyzeCode()">
                Analyze Current File
            </button>
        </div>
        
        <div id="insights" class="insights" style="display: none;">
            <h3>Analysis Results</h3>
            <div id="results">Analyzing...</div>
        </div>
        
        <script>
            const vscode = acquireVsCodeApi();
            
            function analyzeCode() {
                document.getElementById('insights').style.display = 'block';
                vscode.postMessage({ command: 'analyze' });
            }
        </script>
    </body>
    </html>
    `;
}

export function deactivate() {
    if (grepalClient) {
        grepalClient.stop();
    }
    console.log('Grepal deactivated');
}
