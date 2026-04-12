const { useState, useRef, useEffect, useCallback } = React;

const STATUS_MESSAGES = {
    uploading: 'Uploading video\u2026',
    detecting: 'Detecting faces & predicting deepfake\u2026',
    processing_video: 'Generating face detection video\u2026',
};

function barClass(s) { return s > 0.8 ? 'bar-green' : s > 0.2 ? 'bar-orange' : 'bar-red'; }
function scoreClass(s) { return s > 0.8 ? 'score-green' : s > 0.2 ? 'score-orange' : 'score-red'; }

/* ── Simple hash-based router ── */
function useHashRoute() {
    const [route, setRoute] = useState(window.location.hash || '#/');
    useEffect(() => {
        const onHash = () => setRoute(window.location.hash || '#/');
        window.addEventListener('hashchange', onHash);
        return () => window.removeEventListener('hashchange', onHash);
    }, []);
    return route;
}

/* ── Navbar ── */
function Navbar({ route }) {
    return (
        <header className="navbar">
            <div className="logo"><span>AI</span>-Deepfake Video Detection</div>
            <nav>
                <a href="#/" className={route === '#/' || route === '' ? 'active' : ''}>Product</a>
                <a href="#/technology" className={route === '#/technology' ? 'active' : ''}>Technology</a>
            </nav>
        </header>
    );
}

/* ── Main App ── */
function App() {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState(null);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const [submitting, setSubmitting] = useState(false);
    const route = useHashRoute();

    return (
        <>
            <Navbar route={route} />
            {route === '#/technology' ? (
                <TechnologyPage />
            ) : (
                <ProductPage
                    file={file} setFile={setFile}
                    status={status} setStatus={setStatus}
                    error={error} setError={setError}
                    result={result} setResult={setResult}
                    submitting={submitting} setSubmitting={setSubmitting}
                />
            )}
        </>
    );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
