import * as vscode from 'vscode';
import { GrepalClient } from './grepalClient';

let grepalClient: GrepalClient;
let statusBarItem: vscode.StatusBarItem;
let debounceTimer: NodeJS.Timeout | undefined;
let lastCodeHash: string = '';

function hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash = hash & hash;
    }
    return hash;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('Grepal activated');

    grepalClient = new GrepalClient();

    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(bug) Grepal: Ready";
    statusBarItem.show();

    const enableCommand = vscode.commands.registerCommand('grepal.enable', async () => {
        vscode.workspace.getConfiguration('grepal').update('enabled', true, true);
        statusBarItem.text = "$(bug) Grepal: Active";
        try {
            await grepalClient.start();
            vscode.window.showInformationMessage('Grepal enabled!');
        } catch (error) {
            vscode.window.showWarningMessage('Grepal enabled but server connection failed.');
        }
    });

    const disableCommand = vscode.commands.registerCommand('grepal.disable', () => {
        vscode.workspace.getConfiguration('grepal').update('enabled', false, true);
        statusBarItem.text = "$(bug) Grepal: Disabled";
        grepalClient.stop();
    });

    const showInsightsCommand = vscode.commands.registerCommand('grepal.showInsights', () => {
        vscode.window.showInformationMessage('Check your code for insights!');
    });

    const textChangeListener = vscode.workspace.onDidChangeTextDocument(async (event) => {
        const config = vscode.workspace.getConfiguration('grepal');
        if (!config.get('enabled')) return;

        if (event.document === vscode.window.activeTextEditor?.document) {
            if (debounceTimer) clearTimeout(debounceTimer);

            debounceTimer = setTimeout(() => {
                const code = event.document.getText();
                const codeHash = hashCode(code).toString();

                if (codeHash !== lastCodeHash) {
                    lastCodeHash = codeHash;
                    analyzeCurrentFile();
                }
            }, 1500);
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
            console.log('Server connection failed on startup');
        });
        statusBarItem.text = "$(bug) Grepal: Active";
    }
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const document = editor.document;
    const code = document.getText();

    if (code.trim().length < 10) return;

    try {
        statusBarItem.text = "$(sync~spin) Grepal: Analyzing...";

        const analysis = await grepalClient.analyzeCode({
            code: code,
            language: document.languageId,
            filename: document.fileName
        });

        if (analysis.issues.length > 0) {
            statusBarItem.text = `$(warning) Grepal: ${analysis.issues.length} issues`;
            showIssues(analysis.issues, analysis.snarkComment);
        } else {
            statusBarItem.text = "$(check) Grepal: Clean!";
            if (Math.random() < 0.2) {
                vscode.window.showInformationMessage(`Grepal: "${analysis.snarkComment}"`);
            }
        }

    } catch (error: any) {
        console.error('Analysis failed:', error);
        statusBarItem.text = "$(error) Grepal: Error";

        if (error?.message?.includes('ECONNREFUSED')) {
            vscode.window.showErrorMessage('Grepal server not running on http://localhost:8000');
        }
    }
}

function showIssues(issues: any[], snarkComment: string) {
    const topIssues = issues.slice(0, 3);
    const issueText = topIssues.map(i =>
        `Line ${i.line}: ${i.message.substring(0, 60)}...`
    ).join('\n');

    const more = issues.length > 3 ? `\n...and ${issues.length - 3} more` : '';

    vscode.window.showWarningMessage(
        `${snarkComment}\n\n${issueText}${more}`,
        'Dismiss'
    );
}

export function deactivate() {
    if (grepalClient) grepalClient.stop();
    console.log('Grepal deactivated');
}
