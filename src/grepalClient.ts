import axios, { AxiosInstance } from 'axios';
import * as vscode from 'vscode';
export interface CodeAnalysisRequest {
    code: string;
    language: string;
    filename: string;
}

export interface BugIssue {
    type: 'syntax' | 'logic' | 'runtime' | 'style';
    severity: 'low' | 'medium' | 'high' | 'critical';
    line: number;
    column?: number;
    message: string;
    suggestion: string;
    snarkLevel: number;
}

export interface CodeAnalysisResponse {
    issues: BugIssue[];
    overallScore: number;
    snarkComment: string;
    suggestions: string[];
}

export class GrepalClient {
    private httpClient: AxiosInstance;
    private serverUrl: string;
    private isConnected: boolean = false;

    constructor() {
        const config = vscode.workspace.getConfiguration('grepal');
        this.serverUrl = config.get('serverUrl', 'http://localhost:8000');
        
        this.httpClient = axios.create({
            baseURL: this.serverUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('grepal.serverUrl')) {
                this.updateServerUrl();
            }
        });
    }

    private updateServerUrl() {
        const config = vscode.workspace.getConfiguration('grepal');
        this.serverUrl = config.get('serverUrl', 'http://localhost:8000');
        this.httpClient.defaults.baseURL = this.serverUrl;
    }

    async start(): Promise<boolean> {
        try {
            console.log(`Connecting to Grepal server at ${this.serverUrl}`);
            
            const response = await this.httpClient.get('/health');
            
            if (response.status === 200) {
                this.isConnected = true;
                console.log('Grepal server connected');
                return true;
            }
        } catch (error) {
            console.error('ERROR: Failed to connect to Grepal server:', error);
            this.isConnected = false;
            
            vscode.window.showErrorMessage(
                'Could not connect to Grepal server. Make sure it is running.',
                'Start Server'
            ).then(selection => {
                if (selection === 'Start Server') {
                    vscode.window.showInformationMessage(
                        'Run: cd server && python main.py'
                    );
                }
            });
        }
        return false;
    }

    stop() {
        this.isConnected = false;
        console.log('Grepal client stopped');
    }

    async analyzeCode(request: CodeAnalysisRequest): Promise<CodeAnalysisResponse> {
        if (!this.isConnected) {
            await this.start();
        }

        if (!this.isConnected) {
            throw new Error('Grepal server not available');
        }

        try {
            const config = vscode.workspace.getConfiguration('grepal');
            const snarkLevel = config.get('snarkLevel', 'medium');

            const response = await this.httpClient.post('/analyze', {
                ...request,
                snarkLevel
            });

            return response.data;
            
        } catch (error: any) {
            console.error('Analysis request failed:', error);
            
            if (error.response?.status === 503) {
                throw new Error('Grepal is overloaded. Too much bad code to handle!');
            } else if (error.response?.status === 401) {
                throw new Error('Cerebras API key not configured');
            } else {
                throw new Error(`Analysis failed: ${error.message}`);
            }
        }
    }

    async getSimilarBugs(code: string, language: string): Promise<any[]> {
        if (!this.isConnected) {
            await this.start();
        }

        try {
            const response = await this.httpClient.post('/similar-bugs', {
                code,
                language
            });

            return response.data.similar_bugs || [];
        } catch (error) {
            console.error('Similar bugs request failed:', error);
            return [];
        }
    }

    async submitBugFix(bugData: any): Promise<boolean> {
        if (!this.isConnected) {
            await this.start();
        }

        try {
            const response = await this.httpClient.post('/submit-bug', bugData);
            return response.status === 200;
        } catch (error) {
            console.error('Bug submission failed:', error);
            return false;
        }
    }

    isServerConnected(): boolean {
        return this.isConnected;
    }

    async getServerStatus(): Promise<any> {
        try {
            const response = await this.httpClient.get('/status');
            return response.data;
        } catch (error) {
            return { status: 'disconnected', error: error };
        }
    }
}
