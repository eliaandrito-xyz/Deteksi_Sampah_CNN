import { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Loader2 } from 'lucide-react';

interface DetectionResult {
  label: string;
  confidence: number;
  detail: Record<string, number>;
}

export default function ImageDetection({ isServerReady }: { isServerReady: boolean }) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false);

  // Start webcam stream
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      setStream(s);
      setStreaming(true);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Gagal mengakses kamera. Pastikan izin kamera sudah diberikan.');
    }
  };

  // Efek untuk menyambungkan stream ke elemen video setelah video muncul di DOM
  useEffect(() => {
    if (streaming && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(e => console.error("Error playing video:", e));
    }
  }, [streaming, stream]);

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const width = videoRef.current.videoWidth;
    const height = videoRef.current.videoHeight;
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, width, height);
      const dataUrl = canvasRef.current.toDataURL('image/jpeg');
      setImageSrc(dataUrl);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = ev => {
        setImageSrc(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const runDetection = async () => {
    if (!imageSrc) return;
    setDetecting(true);
    setResult(null);

    try {
      // Manual base64 to blob conversion
      const base64Data = imageSrc.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/jpeg' });
      
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');

      // Panggil backend Flask
      const apiResponse = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.error || 'Gagal menghubungi server backend');
      }

      const data = await apiResponse.json();
      setResult(data);
    } catch (err: any) {
      console.error('Detection error:', err);
      alert('Error: ' + err.message);
    } finally {
      setDetecting(false);
    }
  };

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  return (
    <div className="space-y-6">
      {!isServerReady && (
        <div className="bg-red-900/20 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <p className="text-xs text-red-400 font-medium">
            Server Backend (server.py) tidak terdeteksi. Silakan jalankan server untuk mengaktifkan fitur deteksi.
          </p>
        </div>
      )}
      {/* Tab selector */}
      <div className="flex gap-4 items-center">
        <button
          onClick={startCamera}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-lg text-slate-300 hover:bg-slate-700 transition"
        >
          <Camera size={18} />
          Kamera Real‑Time
        </button>
        <label className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-lg text-slate-300 cursor-pointer hover:bg-slate-700 transition">
          <Upload size={18} />
          Unggah Gambar
          <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
        </label>
      </div>

      {/* Video preview */}
      {streaming && (
        <div className="relative w-full max-w-md mx-auto">
          <video ref={videoRef} className="w-full rounded-lg" />
          <button
            onClick={capturePhoto}
            className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-cyan-600 text-white px-4 py-1 rounded"
          >
            Ambil Foto
          </button>
        </div>
      )}

      {/* Image preview */}
      {imageSrc && (
        <div className="flex flex-col items-center gap-3">
          <img src={imageSrc} alt="Selected" className="max-w-xs rounded-lg shadow-lg" />
          <button
            onClick={runDetection}
            disabled={detecting || !isServerReady}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              (detecting || !isServerReady) ? 'bg-slate-600 cursor-not-allowed opacity-50' : 'bg-cyan-600 hover:bg-cyan-500'
            } text-white`}
          >
            {detecting ? <Loader2 className="animate-spin inline-block mr-2" size={16} /> : 'Deteksi'}
          </button>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-4 p-5 bg-slate-800 rounded-xl border border-slate-700 space-y-4">
          <div className="text-center">
            <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-1">Hasil Prediksi</p>
            <p className="text-2xl font-bold text-cyan-400">{result.label}</p>
            <p className="text-sm text-slate-400">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          </div>
          
          <div className="space-y-2">
            {Object.entries(result.detail).map(([cls, prob]) => (
              <div key={cls} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">{cls}</span>
                  <span className="text-slate-400 font-mono">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${cls === result.label ? 'bg-cyan-500' : 'bg-slate-600'}`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
