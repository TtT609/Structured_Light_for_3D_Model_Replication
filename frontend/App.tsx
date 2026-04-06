import React, { useState, useEffect, useRef, useCallback } from 'react';
import { CameraDevice, ConnectionStatus, ServerCommand } from './types';
import { Camera, RefreshCw, Settings, Wifi, WifiOff, Sliders, Zap, ZapOff, Focus, Sun, Search, Timer } from 'lucide-react';

const POLLING_INTERVAL_MS = 500;
const DEFAULT_SERVER_URL = "http://192.168.42.129:5000"; 

interface CameraCapabilities {
  zoom?: { min: number, max: number, step: number };
  focusDistance?: { min: number, max: number, step: number };
  exposureCompensation?: { min: number, max: number, step: number };
  iso?: { min: number, max: number, step: number };
  exposureTime?: { min: number, max: number, step: number };
  exposureMode?: string[];
  torch?: boolean;
}

interface CameraSettings {
  zoom: number;
  focusMode: 'continuous' | 'manual';
  exposureMode: 'continuous' | 'manual';
  focusDistance: number;
  exposureCompensation: number;
  iso: number;
  exposureTime: number;
  torch: boolean;
}

export default function App() {
  // --- STATE ---
  const [serverUrl, setServerUrl] = useState<string>(() => localStorage.getItem('sl_server_url') || DEFAULT_SERVER_URL);
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [logs, setLogs] = useState<string[]>([]);
  const [lastPhoto, setLastPhoto] = useState<string | null>(null);

  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState<string>('');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capabilities, setCapabilities] = useState<CameraCapabilities>({});
  
  const [settings, setSettings] = useState<CameraSettings>({
    zoom: 1,
    focusMode: 'continuous',
    exposureMode: 'continuous',
    focusDistance: 0,
    exposureCompensation: 0,
    iso: 100,
    exposureTime: 10,
    torch: false
  });

  const [showSettings, setShowSettings] = useState(false);
  const [showProMode, setShowProMode] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastProcessedIdRef = useRef<string>('');
  const streamRef = useRef<MediaStream | null>(null);

  const addLog = (msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 5));
  };

  const formatShutterSpeed = (ms: number) => {
    if (!ms) return '';
    if (ms >= 1000) return `${(ms/1000).toFixed(1)}s`;
    return `1/${Math.round(1000 / ms)}s`;
  };

  // 1. INIT
  const getCameras = useCallback(async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices
        .filter(device => device.kind === 'videoinput')
        .map(d => ({ 
            deviceId: d.deviceId, 
            label: d.label || `Camera ${d.deviceId.slice(0,5)}...` 
        }));
      setDevices(videoDevices);
      addLog(`Found ${videoDevices.length} cameras`);
      if (videoDevices.length > 0 && !activeDeviceId) {
        const backCam = videoDevices.find(d => d.label.toLowerCase().includes('back') || d.label.toLowerCase().includes('0')) || videoDevices[0];
        setActiveDeviceId(backCam.deviceId);
      }
    } catch (err: any) {
      addLog(`Cam Init Error: ${err.message}`);
    }
  }, [activeDeviceId]);

  // 2. STREAM & CAPS
  useEffect(() => {
    if (!activeDeviceId) return;
    const startStream = async () => {
      try {
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());

        addLog("Starting stream...");
        const newStream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: activeDeviceId },
            width: { ideal: 3840 },
            height: { ideal: 2160 },
          } as any
        });

        if (videoRef.current) videoRef.current.srcObject = newStream;
        setStream(newStream);
        streamRef.current = newStream;

        const track = newStream.getVideoTracks()[0];
        const caps: any = track.getCapabilities ? track.getCapabilities() : {};
        console.log("Caps:", caps);
        
        setCapabilities({
          zoom: caps.zoom,
          focusDistance: caps.focusDistance,
          exposureCompensation: caps.exposureCompensation,
          iso: caps.iso,
          exposureTime: caps.exposureTime,
          exposureMode: caps.exposureMode,
          torch: caps.torch
        });

        setSettings(prev => ({
          ...prev,
          zoom: caps.zoom ? caps.zoom.min : 1,
          focusDistance: caps.focusDistance ? caps.focusDistance.min : 0,
          exposureCompensation: caps.exposureCompensation ? 0 : 0, 
          iso: caps.iso ? caps.iso.min : 100,
          exposureTime: caps.exposureTime ? 10 : 10,
          focusMode: 'continuous',
          exposureMode: 'continuous' 
        }));
      } catch (err: any) {
        addLog(`Stream Error: ${err.message}`);
      }
    };
    startStream();
    return () => { if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop()); };
  }, [activeDeviceId]);

  // 3. APPLY SETTINGS
  const applyConstraint = async (newSettings: CameraSettings) => {
    if (!streamRef.current) return;
    const track = streamRef.current.getVideoTracks()[0];
    const advancedSet: any = {};

    // Focus
    if (newSettings.focusMode === 'manual' && capabilities.focusDistance) {
        advancedSet.focusMode = 'manual';
        advancedSet.focusDistance = newSettings.focusDistance;
    } else {
        advancedSet.focusMode = 'continuous';
    }

    // Exposure
    if (newSettings.exposureMode === 'manual') {
        advancedSet.exposureMode = 'manual'; 
        if (capabilities.iso) advancedSet.iso = newSettings.iso;
        if (capabilities.exposureTime) advancedSet.exposureTime = newSettings.exposureTime;
    } else {
        advancedSet.exposureMode = 'continuous';
        if (capabilities.exposureCompensation) advancedSet.exposureCompensation = newSettings.exposureCompensation;
    }

    // Zoom/Torch
    if (capabilities.zoom) advancedSet.zoom = newSettings.zoom;
    if (capabilities.torch !== undefined) advancedSet.torch = newSettings.torch;

    try {
        await track.applyConstraints({ advanced: [advancedSet] } as any);
    } catch (e: any) {
        console.warn("Constraint failed:", e);
    }
  };

  const updateSetting = (key: keyof CameraSettings, value: any) => {
    setSettings(prev => {
        const next = { ...prev, [key]: value };
        if (key === 'iso' || key === 'exposureTime') {
            next.exposureMode = 'manual';
            next.exposureCompensation = 0;
        }
        if (key === 'exposureCompensation') {
            next.exposureMode = 'continuous';
        }
        applyConstraint(next);
        return next;
    });
  };

  // 4. POLL & CAPTURE
  useEffect(() => {
    const pollServer = async () => {
      if (status === ConnectionStatus.UPLOADING) return;
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        const res = await fetch(`${serverUrl}/poll_command`, { signal: controller.signal, mode: 'cors' });
        clearTimeout(timeoutId);

        if (res.ok) {
          const data: ServerCommand = await res.json();
          if (status === ConnectionStatus.DISCONNECTED || status === ConnectionStatus.ERROR) setStatus(ConnectionStatus.CONNECTED);
          if (data.action === 'capture' && data.id && data.id !== lastProcessedIdRef.current) {
            lastProcessedIdRef.current = data.id;
            handleCapture();
          }
        } else {
          setStatus(ConnectionStatus.ERROR);
        }
      } catch (err) {
        if (status === ConnectionStatus.CONNECTED) setStatus(ConnectionStatus.ERROR);
      }
    };
    const interval = setInterval(pollServer, POLLING_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [serverUrl, status]);

  const handleCapture = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    setStatus(ConnectionStatus.CAPTURING);
    addLog("Capturing...");
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setStatus(ConnectionStatus.UPLOADING);
    canvas.toBlob(async (blob) => {
      if (!blob) { setStatus(ConnectionStatus.ERROR); return; }
      setLastPhoto(URL.createObjectURL(blob));
      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');
      try {
        const res = await fetch(`${serverUrl}/upload`, { method: 'POST', body: formData, mode: 'cors' });
        if (res.ok) { addLog("Uploaded."); setStatus(ConnectionStatus.CONNECTED); } 
        else { throw new Error("Rejected"); }
      } catch (err: any) {
        addLog(`Upload Err: ${err.message}`);
        setStatus(ConnectionStatus.ERROR);
      }
    }, 'image/jpeg', 1.0); 
  };

  const handleSaveSettings = () => {
    localStorage.setItem('sl_server_url', serverUrl);
    setShowSettings(false);
  };

  return (
    <div className="flex flex-col h-screen bg-black text-white overflow-hidden">
      {/* Top Bar */}
      <div className="flex items-center justify-between p-3 bg-slate-900 border-b border-slate-700 z-20">
        <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-blue-400" />
            <span className="font-bold text-sm">Project SL</span>
        </div>
        <div className="flex gap-2">
            <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded-full border ${
                status === ConnectionStatus.CONNECTED ? 'bg-green-950 border-green-700 text-green-400' : 
                status === ConnectionStatus.UPLOADING ? 'bg-yellow-950 border-yellow-700 text-yellow-400' :
                'bg-red-950 border-red-700 text-red-400'
            }`}>
                {status === ConnectionStatus.CONNECTED ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                {status === ConnectionStatus.CONNECTED ? 'LINKED' : status}
            </div>
            <button onClick={() => setShowProMode(!showProMode)} className={`p-1.5 rounded border ${showProMode ? 'bg-blue-600 border-blue-400 text-white' : 'bg-slate-800 border-slate-600 text-slate-300'}`}>
                <Sliders className="w-4 h-4" />
            </button>
            <button onClick={() => setShowSettings(!showSettings)} className="p-1.5 rounded bg-slate-800 border border-slate-600 text-slate-300">
                <Settings className="w-4 h-4" />
            </button>
        </div>
      </div>

      {/* Viewport */}
      <div className="flex-1 relative bg-black flex items-center justify-center">
        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-contain" />
        <canvas ref={canvasRef} className="hidden" />
        
        {status === ConnectionStatus.UPLOADING && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-10 backdrop-blur-sm">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-white"></div>
            </div>
        )}
        
        <div className="absolute bottom-24 left-2 z-10">
            {logs.map((log, i) => (
                <div key={i} className="text-[10px] text-green-400 bg-black/50 px-1.5 py-0.5 rounded mb-0.5 backdrop-blur-md border border-white/10 w-fit">{log}</div>
            ))}
        </div>
        
        {lastPhoto && (
            <div className="absolute bottom-24 right-2 z-10 w-16 h-16 border border-slate-500 bg-black rounded shadow-lg">
                <img src={lastPhoto} alt="" className="w-full h-full object-cover opacity-80" />
            </div>
        )}

        {/* PRO MODE */}
        {showProMode && (
            <div className="absolute bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur border-t border-slate-700 p-4 pb-24 z-20 flex flex-col gap-4 animate-in slide-in-from-bottom-10 transition-all">
                
                {/* Zoom */}
                {capabilities.zoom && (
                    <div className="flex items-center gap-3">
                        <Search className="w-5 h-5 text-slate-400" />
                        <span className="text-xs text-slate-400 w-12">ZOOM</span>
                        <input type="range" min={capabilities.zoom.min} max={capabilities.zoom.max} step={0.1} value={settings.zoom}
                            onChange={(e) => updateSetting('zoom', parseFloat(e.target.value))}
                            className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                        <span className="text-xs w-8 text-right">{settings.zoom.toFixed(1)}x</span>
                    </div>
                )}

                {/* Focus */}
                <div className="flex items-center gap-3">
                    <Focus className="w-5 h-5 text-slate-400" />
                    <button onClick={() => updateSetting('focusMode', settings.focusMode === 'continuous' ? 'manual' : 'continuous')}
                        className={`text-xs px-2 py-1 rounded border ${settings.focusMode === 'continuous' ? 'bg-green-600 border-green-400 text-white' : 'bg-slate-700 border-slate-500 text-slate-300'}`}>
                        {settings.focusMode === 'continuous' ? 'AF' : 'MF'}
                    </button>
                    {settings.focusMode === 'manual' && capabilities.focusDistance ? (
                        <input type="range" min={capabilities.focusDistance.min} max={capabilities.focusDistance.max} step={capabilities.focusDistance.step} value={settings.focusDistance}
                            onChange={(e) => updateSetting('focusDistance', parseFloat(e.target.value))}
                            className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-yellow-500" />
                    ) : <div className="flex-1 text-xs text-slate-500 italic ml-2">Auto Focus Active</div>}
                </div>

                {/* Auto Exposure */}
                {capabilities.exposureCompensation && (
                    <div className={`flex items-center gap-3 transition-opacity ${settings.exposureMode === 'manual' ? 'opacity-30 pointer-events-none' : 'opacity-100'}`}>
                        <Sun className="w-5 h-5 text-slate-400" />
                        <span className="text-xs text-slate-400 w-12">EV</span>
                        <input type="range" min={capabilities.exposureCompensation.min} max={capabilities.exposureCompensation.max} step={capabilities.exposureCompensation.step} value={settings.exposureCompensation}
                            onChange={(e) => updateSetting('exposureCompensation', parseFloat(e.target.value))}
                            className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-white" />
                        <span className="text-xs w-8 text-right">{settings.exposureCompensation > 0 ? '+' : ''}{settings.exposureCompensation}</span>
                    </div>
                )}

                {/* Manual Exposure */}
                <div className={`flex flex-col gap-2 p-2 rounded border border-white/10 ${settings.exposureMode === 'continuous' ? 'opacity-50' : 'bg-white/5'}`}>
                    <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Manual Exposure</div>
                    
                    {capabilities.iso && (
                        <div className="flex items-center gap-3">
                            <span className="text-[10px] text-slate-400 w-12">ISO</span>
                            <input type="range" min={capabilities.iso.min} max={capabilities.iso.max} step={capabilities.iso.step} value={settings.iso}
                                onInput={(e) => updateSetting('iso', parseFloat((e.target as HTMLInputElement).value))}
                                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500" />
                            
                            {/* NEW: Input Box for ISO */}
                            <input 
                                type="number" 
                                value={settings.iso}
                                onChange={(e) => updateSetting('iso', parseFloat(e.target.value))}
                                className="w-12 bg-transparent text-right text-xs border-b border-slate-500 focus:border-purple-500 outline-none appearance-none font-mono"
                            />
                        </div>
                    )}

                    {capabilities.exposureTime && (
                        <div className="flex items-center gap-3">
                            <Timer className="w-3 h-3 text-slate-400 ml-1 mr-[-4px]" />
                            <span className="text-[10px] text-slate-400 w-11">TIME</span>
                            <input type="range" min={capabilities.exposureTime.min} max={capabilities.exposureTime.max} step={capabilities.exposureTime.step} value={settings.exposureTime}
                                onInput={(e) => updateSetting('exposureTime', parseFloat((e.target as HTMLInputElement).value))}
                                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500" />
                            
                            {/* NEW: Input Box for Time + Fraction Label */}
                            <div className="flex items-center gap-1">
                                <input 
                                    type="number" 
                                    value={settings.exposureTime}
                                    onChange={(e) => updateSetting('exposureTime', parseFloat(e.target.value))}
                                    className="w-10 bg-transparent text-right text-xs border-b border-slate-500 focus:border-cyan-500 outline-none appearance-none font-mono"
                                />
                                <span className="text-[9px] text-slate-500 w-8 text-right font-mono">{formatShutterSpeed(settings.exposureTime)}</span>
                            </div>
                        </div>
                    )}
                </div>

                {capabilities.torch !== undefined && (
                        <button onClick={() => updateSetting('torch', !settings.torch)}
                        className={`flex flex-col items-center justify-center w-full h-10 rounded border ${settings.torch ? 'bg-yellow-500 border-yellow-300 text-black' : 'bg-slate-800 border-slate-600 text-slate-400'}`}>
                        {settings.torch ? <Zap className="w-4 h-4" /> : <ZapOff className="w-4 h-4" />}
                        <span className="text-[9px] font-bold mt-1">FLASHLIGHT</span>
                        </button>
                )}
            </div>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="absolute inset-0 z-30 bg-black/80 backdrop-blur flex items-center justify-center p-6">
            <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-sm shadow-2xl">
                <h2 className="text-lg font-bold mb-4 flex items-center gap-2"><Settings className="w-5 h-5" /> App Settings</h2>
                <div className="space-y-4">
                    <div>
                        <label className="text-xs text-slate-400 uppercase font-bold tracking-wider">Server URL</label>
                        <input type="text" value={serverUrl} onChange={(e) => setServerUrl(e.target.value)}
                            className="w-full mt-1 bg-black border border-slate-700 p-3 rounded text-sm text-white focus:border-blue-500 outline-none" placeholder="http://192.168.x.x:5000" />
                    </div>
                    <div>
                        <label className="text-xs text-slate-400 uppercase font-bold tracking-wider">Physical Camera ID</label>
                        <select value={activeDeviceId} onChange={(e) => { setActiveDeviceId(e.target.value); setShowSettings(false); }}
                            className="w-full mt-1 bg-black border border-slate-700 p-3 rounded text-sm text-white outline-none">
                            {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label}</option>)}
                        </select>
                    </div>
                    <div className="flex gap-2 pt-2">
                        <button onClick={getCameras} className="flex-1 py-3 bg-slate-800 rounded hover:bg-slate-700 text-sm font-medium flex items-center justify-center gap-2"><RefreshCw className="w-4 h-4" /> Rescan</button>
                        <button onClick={handleSaveSettings} className="flex-1 py-3 bg-blue-600 rounded hover:bg-blue-500 text-sm font-medium">Save & Close</button>
                    </div>
                </div>
            </div>
        </div>
      )}

      {/* Trigger */}
      <div className="absolute bottom-6 left-0 right-0 flex justify-center pointer-events-none z-30">
        <button onClick={handleCapture}
          className="pointer-events-auto bg-red-600/90 hover:bg-red-500 text-white w-20 h-20 rounded-full flex items-center justify-center shadow-lg border-4 border-white/20 ring-2 ring-red-600 transition-transform active:scale-95">
          <div className="w-16 h-16 rounded-full border-2 border-white/40" />
        </button>
      </div>
    </div>
  );
}