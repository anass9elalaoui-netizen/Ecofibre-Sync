import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Leaf, Cpu, AlertCircle, Activity, ChevronRight, Zap, RefreshCw, UploadCloud, FileText, CheckCircle2, Database } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ── Types ──────────────────────────────────────────────────────────────────
interface ResolutionResponse {
  equipment: string;
  error_codes: string[];
  error_description: string;
  context_used: boolean;
  procedure: string;
  rse_score: number;
  processing_time_seconds: number;
  pipeline_steps: string[];
}

interface IngestStatus {
  status: 'PENDING' | 'PARSING' | 'EMBEDDING' | 'COMPLETED' | 'FAILED';
  chunk_count?: number;
  error?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL 
  ? (import.meta.env.VITE_API_URL.startsWith('http') ? import.meta.env.VITE_API_URL : `https://${import.meta.env.VITE_API_URL}`)
  : 'http://localhost:8000';

// ── Main App Component ──────────────────────────────────────────────────────
function App() {
  const [activeTab, setActiveTab] = useState<'diagnostic' | 'knowledge'>('diagnostic');

  // Diagnostic State
  const [reportText, setReportText] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ResolutionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Knowledge Base State
  const [isDragging, setIsDragging] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<IngestStatus | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Diagnostic Logic ──────────────────────────────────────────────────────
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string); // Base64 string
        setPreview(URL.createObjectURL(file)); // URL locale pour la prévisualisation
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!reportText.trim() && !image) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/resolution/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report: reportText, image: image }),
      });

      if (!response.ok) throw new Error(`Erreur API: ${response.statusText}`);
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Une erreur est survenue.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
    if (score >= 60) return 'text-green-400 bg-green-400/10 border-green-400/20';
    if (score >= 40) return 'text-amber-400 bg-amber-400/10 border-amber-400/20';
    return 'text-red-400 bg-red-400/10 border-red-400/20';
  };

  // ── Knowledge Base Logic ──────────────────────────────────────────────────
  const startPolling = (documentId: string) => {
    if (pollingInterval.current) clearInterval(pollingInterval.current);

    pollingInterval.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/ingest/${documentId}/status`);
        if (!response.ok) throw new Error('Polling failed');
        const data: IngestStatus = await response.json();
        // Normalize status to uppercase (backend sends lowercase enum values)
        data.status = data.status.toUpperCase() as IngestStatus['status'];
        
        setUploadStatus(data);
        
        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          if (pollingInterval.current) clearInterval(pollingInterval.current);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2000); // Poll every 2s
  };

  const handleFileUpload = async (file: File) => {
    setUploadFile(file);
    setUploadStatus({ status: 'PENDING' });
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ingest/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Upload échoué: ${response.statusText}`);
      const data = await response.json();
      
      // Start polling with the returned document_id
      if (data.document_id) {
        startPolling(data.document_id);
      }
    } catch (err) {
      setUploadStatus({ 
        status: 'FAILED', 
        error: err instanceof Error ? err.message : 'Erreur réseau' 
      });
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const onDragLeave = () => setIsDragging(false);
  
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  useEffect(() => {
    return () => {
      if (pollingInterval.current) clearInterval(pollingInterval.current);
    };
  }, []);

  const getStepStatus = (stepIndex: number, currentStatus: string | undefined) => {
    const statuses = ['PENDING', 'PARSING', 'EMBEDDING', 'COMPLETED'];
    const currentIndex = statuses.indexOf(currentStatus || 'PENDING');
    
    if (currentStatus === 'FAILED') return stepIndex === currentIndex ? 'error' : 'waiting';
    if (currentIndex > stepIndex) return 'completed';
    if (currentIndex === stepIndex) return 'current';
    return 'waiting';
  };

  return (
    <div className="min-h-screen bg-slate-900 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-900 to-eco-primary/10 relative overflow-hidden">
      
      {/* Decorative background */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-eco-primary/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] rounded-full bg-blue-500/5 blur-[100px] pointer-events-none" />

      <main className="max-w-5xl mx-auto px-6 py-12 relative z-10">
        
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 mb-12">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-eco-primary to-emerald-600 flex items-center justify-center shadow-lg shadow-eco-primary/20">
              <Leaf className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
                EcoFibre <span className="text-transparent bg-clip-text bg-gradient-to-r from-eco-primary to-emerald-400">AI</span>
              </h1>
              <p className="text-slate-400 text-sm mt-1">Maintenance FTTH Numérique Responsable</p>
            </div>
          </div>
          
          {/* Tabs Navigation */}
          <div className="flex p-1 bg-slate-800/50 rounded-xl border border-slate-700/50 backdrop-blur-sm">
            <button
              onClick={() => setActiveTab('diagnostic')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'diagnostic' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/30"
              )}
            >
              <Zap className="w-4 h-4" />
              Diagnostic IA
            </button>
            <button
              onClick={() => setActiveTab('knowledge')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                activeTab === 'knowledge' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/30"
              )}
            >
              <Database className="w-4 h-4" />
              Base de Connaissances
            </button>
          </div>
        </header>

        {/* ── Tab Content: Diagnostic ── */}
        {activeTab === 'diagnostic' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-300">
            {/* Input Panel (Left) */}
            <div className="lg:col-span-5 space-y-6">
              <div className="glass-panel p-6 transition-all duration-300 hover:border-slate-600/50">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-eco-primary" />
                  Rapport d'intervention
                </h2>
                <div className="relative group">
                  <textarea
                    value={reportText}
                    onChange={(e) => setReportText(e.target.value)}
                    placeholder="Ex: Client sans internet depuis ce matin. ONT avec voyant LOS rouge fixe..."
                    className="w-full h-48 bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-slate-300 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-eco-primary/50 focus:border-eco-primary/50 transition-all resize-none shadow-inner"
                  />
                </div>
                
                {/* Image Upload Zone */}
                <div className="mt-4 flex items-center space-x-4">
                  <label className="cursor-pointer bg-slate-800 hover:bg-slate-700 text-white py-2 px-4 rounded-lg flex items-center border border-slate-600/50 transition-colors">
                    <svg className="w-5 h-5 mr-2 text-eco-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    Ajouter une photo
                    <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                  </label>
                  
                  {/* Prévisualisation */}
                  {preview && (
                    <div className="relative group">
                      <img src={preview} alt="Aperçu" className="h-12 w-12 object-cover rounded border border-slate-500 shadow-sm" />
                      <button 
                        onClick={() => {setImage(null); setPreview(null)}} 
                        className="absolute -top-2 -right-2 bg-red-500/90 hover:bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs shadow-md transition-all"
                      >
                        X
                      </button>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || (!reportText.trim() && !image)}
                  className="mt-6 w-full relative group overflow-hidden rounded-xl p-[1px] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span className="absolute inset-0 bg-gradient-to-r from-eco-primary via-emerald-400 to-eco-secondary opacity-70 group-hover:opacity-100 transition-opacity duration-300" />
                  <div className={cn(
                    "relative flex items-center justify-center gap-2 bg-slate-900 px-6 py-3.5 rounded-xl transition-all duration-300",
                    isAnalyzing ? "opacity-90" : "group-hover:bg-slate-900/50"
                  )}>
                    {isAnalyzing ? (
                      <>
                        <RefreshCw className="w-5 h-5 text-eco-primary animate-spin" />
                        <span className="font-medium text-white">Analyse LangGraph en cours...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5 text-eco-primary group-hover:text-white transition-colors" />
                        <span className="font-medium text-white">Analyser avec EcoFibre AI</span>
                      </>
                    )}
                  </div>
                </button>
              </div>
            </div>

            {/* Result Panel (Right) */}
            <div className="lg:col-span-7">
              {error ? (
                <div className="glass-panel p-6 border-red-500/20 bg-red-500/5 flex items-start gap-4">
                  <AlertCircle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-red-400 font-medium">Erreur lors de l'analyse</h3>
                    <p className="text-red-300/70 text-sm mt-1">{error}</p>
                  </div>
                </div>
              ) : result ? (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  {/* Extracted Data Badges */}
                  <div className="flex flex-wrap gap-4">
                    <div className="glass-panel px-4 py-2 flex items-center gap-3 border-l-2 border-l-blue-500">
                      <Cpu className="w-4 h-4 text-blue-400" />
                      <div>
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Équipement</p>
                        <p className="text-sm font-medium text-slate-200">{result.equipment}</p>
                      </div>
                    </div>
                    
                    {result.error_codes.length > 0 && (
                      <div className="glass-panel px-4 py-2 flex items-center gap-3 border-l-2 border-l-amber-500">
                        <AlertCircle className="w-4 h-4 text-amber-400" />
                        <div>
                          <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Codes Erreur</p>
                          <div className="flex gap-1.5 mt-0.5">
                            {result.error_codes.map(code => (
                              <span key={code} className="text-xs px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                                {code}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    <div className={cn("glass-panel px-4 py-2 flex items-center gap-3 border-l-2", getScoreColor(result.rse_score).replace('text-', 'border-l-'))}>
                      <Leaf className={cn("w-4 h-4", getScoreColor(result.rse_score).split(' ')[0])} />
                      <div>
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Score RSE</p>
                        <p className={cn("text-sm font-bold", getScoreColor(result.rse_score).split(' ')[0])}>
                          {result.rse_score} / 100
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Markdown Procedure */}
                  <div className="glass-panel p-8 relative">
                    <div className="absolute top-0 right-0 p-4 opacity-50 flex items-center gap-2 text-xs text-slate-500">
                      <span>Généré en {result.processing_time_seconds}s</span>
                      {result.context_used && (
                        <span className="px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700">RAG Context: Oui</span>
                      )}
                    </div>
                    <div className="markdown-body">
                      <ReactMarkdown>{result.procedure}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full min-h-[400px] glass-panel border-dashed border-slate-700/50 flex flex-col items-center justify-center text-slate-500">
                  <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-4">
                    <ChevronRight className="w-8 h-8 text-slate-600" />
                  </div>
                  <p>Saisissez un rapport d'intervention pour obtenir</p>
                  <p>une procédure de résolution éco-responsable.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Tab Content: Knowledge Base Upload ── */}
        {activeTab === 'knowledge' && (
          <div className="max-w-2xl mx-auto animate-in fade-in duration-300 space-y-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-white mb-2">Enrichir l'Intelligence IA</h2>
              <p className="text-slate-400">
                Glissez-déposez des manuels constructeurs, normes ou rapports d'incidents pour alimenter la base vectorielle (pgvector).
              </p>
            </div>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "glass-panel border-2 border-dashed p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300",
                isDragging 
                  ? "border-eco-primary bg-eco-primary/5" 
                  : "border-slate-600 hover:border-slate-500 hover:bg-slate-800/80"
              )}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
                className="hidden" 
                accept=".txt,.pdf,.md"
              />
              <div className={cn(
                "w-20 h-20 rounded-full flex items-center justify-center mb-6 transition-all duration-300",
                isDragging ? "bg-eco-primary/20 scale-110" : "bg-slate-800"
              )}>
                <UploadCloud className={cn("w-10 h-10 transition-colors", isDragging ? "text-eco-primary" : "text-slate-400")} />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Glissez un fichier ici</h3>
              <p className="text-slate-400 text-sm">PDF, TXT, MD supportés (max 10MB)</p>
            </div>

            {/* Upload Progress Area */}
            {uploadFile && uploadStatus && (
              <div className="glass-panel p-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{uploadFile.name}</p>
                    <p className="text-xs text-slate-400">{(uploadFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                  {uploadStatus.status === 'COMPLETED' && (
                    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      Terminé ({uploadStatus.chunk_count} chunks)
                    </span>
                  )}
                  {uploadStatus.status === 'FAILED' && (
                    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">
                      Échec
                    </span>
                  )}
                </div>

                {/* Progress Steps */}
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-700/50"></div>
                  
                  {[
                    { title: "En attente", desc: "Mise en file d'attente Celery", step: 0 },
                    { title: "Parsing", desc: "Extraction du texte et découpage en chunks", step: 1 },
                    { title: "Vectorisation", desc: "Génération des embeddings via Google AI", step: 2 },
                    { title: "Indexation", desc: "Sauvegarde dans pgvector", step: 3 }
                  ].map((item, idx) => {
                    const status = getStepStatus(item.step, uploadStatus.status);
                    return (
                      <div key={idx} className="relative pl-10 py-3 flex items-start gap-4">
                        <div className={cn(
                          "absolute left-3 -translate-x-1/2 w-3 h-3 rounded-full border-2 transition-all duration-500 z-10",
                          status === 'completed' ? "bg-emerald-500 border-emerald-500" :
                          status === 'current' ? "bg-slate-900 border-eco-primary ring-4 ring-eco-primary/20" :
                          status === 'error' ? "bg-red-500 border-red-500" :
                          "bg-slate-900 border-slate-600"
                        )} />
                        <div>
                          <h4 className={cn("text-sm font-medium", 
                            status === 'waiting' ? "text-slate-500" :
                            status === 'error' ? "text-red-400" : "text-white"
                          )}>
                            {item.title}
                          </h4>
                          <p className="text-xs text-slate-400 mt-0.5">{item.desc}</p>
                        </div>
                        {status === 'current' && <RefreshCw className="w-4 h-4 text-eco-primary animate-spin ml-auto" />}
                        {status === 'completed' && <CheckCircle2 className="w-4 h-4 text-emerald-500 ml-auto" />}
                      </div>
                    );
                  })}
                </div>

                {uploadStatus.error && (
                  <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                    {uploadStatus.error}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
