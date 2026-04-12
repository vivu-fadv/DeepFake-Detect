/* ── Upload Area ── */
function UploadArea({ file, onFileChange }) {
    const inputRef = React.useRef();
    return (
        <div
            className={`upload-area${file ? ' has-file' : ''}`}
            onClick={() => inputRef.current.click()}
        >
            <div className="upload-text">
                Drop a video here or click to upload &mdash; MP4, AVI, MOV, max 200 MB
            </div>
            {file && <div className="file-name">{file.name}</div>}
            <input
                ref={inputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.wmv"
                onChange={e => { onFileChange(e.target.files[0] || null); e.target.value = ''; }}
            />
        </div>
    );
}

/* ── Status Spinner ── */
function StatusIndicator({ status }) {
    if (!status || status === 'done') return null;
    return (
        <>
            <div className="spinner" />
            <p className="processing-text">{STATUS_MESSAGES[status] || 'Processing\u2026'}</p>
        </>
    );
}

/* ── Video Comparison ── */
const VideoComparison = React.memo(function VideoComparison({ file, processedUrl, isProcessing }) {
    const videoRef = React.useRef(null);
    const [localUrl, setLocalUrl] = React.useState(null);

    React.useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setLocalUrl(url);
            return () => URL.revokeObjectURL(url);
        }
        setLocalUrl(null);
    }, [file]);

    if (!localUrl) return null;
    const showProcessed = processedUrl || isProcessing;
    return (
        <section className="video-compare">
            <h2>{showProcessed ? 'Face Detection' : 'Uploaded Video'}</h2>
            <div className={`compare-grid${showProcessed ? '' : ' single'}`}>
                <div className="compare-item">
                    <video ref={videoRef} controls src={localUrl} />
                    <div className="compare-label original">Original</div>
                </div>
                {showProcessed && (
                    <div className="compare-item">
                        {processedUrl
                            ? <video controls src={processedUrl} />
                            : <div className="preview-placeholder"><div className="spinner" /></div>
                        }
                        <div className="compare-label detected">
                            {processedUrl ? 'Detected Faces' : 'Generating\u2026'}
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
});

/* ── Face Row ── */
function FaceRow({ face, index }) {
    const pct = (face.score * 100).toFixed(2);
    const w = (face.score * 100).toFixed(1);
    return (
        <div className="face-row">
            <img className="face-thumb" src={`data:image/png;base64,${face.thumbnail}`} alt={`Face ${index + 1}`} />
            <div className="face-info">
                <div className="face-bar-track">
                    <div className={`face-bar-fill ${barClass(face.score)}`} style={{ width: `${w}%` }} />
                </div>
                <div className={`face-score ${scoreClass(face.score)}`}>{pct}% authentic</div>
            </div>
        </div>
    );
}

/* ── Results Panel ── */
function ResultsPanel({ data }) {
    if (!data || !data.result) return null;
    const cls = data.result.toLowerCase();
    return (
        <section className="results-section">
            <div className="results-panel">
                <h2>Results</h2>
                <p className="results-hint">
                    Authenticity score: likelihood the face is real.{' '}
                    <span className="score-red">Red &lt;20%</span>,{' '}
                    <span className="score-orange">Orange 20-80%</span>,{' '}
                    <span className="score-green">Green &gt;80%</span>.
                </p>
                <div className={`overall-result ${cls}`}>
                    <div className="overall-label">{data.result}</div>
                    <div className="overall-details">
                        Confidence: {data.confidence}%<br />
                        Model Score: {data.score}<br />
                        Faces Analyzed: {data.num_faces}
                    </div>
                </div>
                {data.faces_detail && data.faces_detail.map((face, i) => (
                    <FaceRow key={i} face={face} index={i} />
                ))}
            </div>
        </section>
    );
}

/* ── Product Page ── */
function ProductPage({ file, setFile, status, setStatus, error, setError, result, setResult, submitting, setSubmitting }) {
    const timerRef = React.useRef(null);

    const reset = () => { setResult(null); setError(null); setStatus(null); };
    const handleFileChange = (f) => { setFile(f); reset(); };

    const pollJob = React.useCallback((jobId) => {
        timerRef.current = setTimeout(async () => {
            try {
                const res = await fetch(`/status/${jobId}`);
                const data = await res.json();
                setStatus(data.status);

                if (data.result) {
                    setResult(prev => ({ ...prev, ...data }));
                }

                if (data.status === 'done') {
                    setSubmitting(false);
                    if (data.error && !data.result) setError(data.error);
                } else {
                    pollJob(jobId);
                }
            } catch {
                setSubmitting(false);
                setStatus(null);
                setError('Connection lost. Please try again.');
            }
        }, 1000);
    }, []);

    React.useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;
        reset();
        setSubmitting(true);
        setStatus('uploading');

        const fd = new FormData();
        fd.append('video', file);

        try {
            const res = await fetch('/predict', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) {
                setError(data.error);
                setSubmitting(false);
                setStatus(null);
            } else {
                pollJob(data.job_id);
            }
        } catch {
            setError('Upload failed. Please try again.');
            setSubmitting(false);
            setStatus(null);
        }
    };

    return (
        <>
            <section className="hero">
                <div className="hero-left">
                    <h1 className="hero-title">AI-Deepfake Video Detection</h1>
                    <p className="hero-desc">
                        Free deepfake detection tool for videos. Upload a video and get
                        per-face authenticity scores in seconds. AI-powered synthetic face detection.
                    </p>

                    <form onSubmit={handleSubmit}>
                        <UploadArea file={file} onFileChange={handleFileChange} />
                        <button type="submit" className="btn" disabled={!file || submitting}>
                            {submitting ? (STATUS_MESSAGES[status] || 'Processing\u2026') : 'Analyze Video'}
                        </button>
                    </form>

                    <StatusIndicator status={submitting ? status : null} />
                    {error && <div className="error-box">{error}</div>}
                </div>

                <div className="hero-right">
                    <img src="/static/images.png" alt="AI Deepfake Detection" className="hero-image" />
                </div>
            </section>

            <VideoComparison
                file={file}
                processedUrl={result?.processed_url}
                isProcessing={submitting}
            />
            <ResultsPanel data={result} />
        </>
    );
}
