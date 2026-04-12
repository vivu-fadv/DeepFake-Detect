/* ── Step Diagram SVG ── */
function StepDiagram({ number, title, color }) {
    return (
        <svg width="80" height="80" viewBox="0 0 80 80" className="step-icon">
            <circle cx="40" cy="40" r="36" fill="none" stroke={color} strokeWidth="2.5" />
            <text x="40" y="46" textAnchor="middle" fill={color} fontSize="28" fontWeight="700">{number}</text>
        </svg>
    );
}

/* ── Pipeline Flow Arrow ── */
function FlowArrow() {
    return (
        <div className="flow-arrow">
            <svg width="40" height="40" viewBox="0 0 40 40">
                <path d="M10 20 L28 20 M22 13 L30 20 L22 27" fill="none" stroke="#6c8cff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
        </div>
    );
}

/* ── Technology Page ── */
function TechnologyPage() {
    const steps = [
        {
            number: 1,
            title: 'Video to Frames',
            color: '#6c8cff',
            description: 'Raw training videos are split into individual image frames. One frame is extracted per second of video using OpenCV. Each frame is automatically scaled based on its resolution to normalize image sizes across the dataset.',
            details: [
                'Reads MP4 videos from the FaceForensics++ dataset',
                'Extracts 1 frame per second (at the video\'s native frame rate)',
                'Auto-scales: 2\u00d7 for small frames (<300px), 0.5\u00d7 for HD, 0.33\u00d7 for Full HD+',
                'Saves frames as individual JPG images organized by video',
            ],
            diagram: (
                <svg viewBox="0 0 400 120" className="step-illustration">
                    <rect x="10" y="15" width="90" height="90" rx="8" fill="#1a1a2e" stroke="#6c8cff" strokeWidth="1.5"/>
                    <polygon points="40,35 40,80 70,57" fill="#6c8cff" opacity="0.8"/>
                    <text x="55" y="112" textAnchor="middle" fill="#888" fontSize="11">Video</text>
                    <path d="M115 60 L155 60" stroke="#6c8cff" strokeWidth="2" markerEnd="url(#arrow1)"/>
                    <defs><marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6c8cff"/></marker></defs>
                    {[0,1,2,3,4].map(i => (
                        <g key={i}>
                            <rect x={165 + i*46} y={20 + (i%2)*15} width="38" height="38" rx="4" fill="#1a1a2e" stroke="#4caf50" strokeWidth="1"/>
                            <text x={184 + i*46} y={44 + (i%2)*15} textAnchor="middle" fill="#4caf50" fontSize="9">F{i+1}</text>
                        </g>
                    ))}
                    <text x="300" y="112" textAnchor="middle" fill="#888" fontSize="11">Extracted Frames (1/sec)</text>
                </svg>
            ),
        },
        {
            number: 2,
            title: 'Face Detection & Cropping',
            color: '#ff9800',
            description: 'MTCNN (Multi-task Cascaded Convolutional Network) scans each extracted frame to detect faces. Detected faces are cropped with a 30% margin around the bounding box to preserve context like hair and jawline, which helps the model detect manipulation artifacts.',
            details: [
                'Uses MTCNN deep learning face detector for accurate face localization',
                'Filters low-confidence detections (>95% threshold for multi-face frames)',
                'Adds 30% margin around each face bounding box',
                'Crops and saves individual face images for training',
            ],
            diagram: (
                <svg viewBox="0 0 400 120" className="step-illustration">
                    <rect x="10" y="10" width="100" height="100" rx="6" fill="#1a1a2e" stroke="#444" strokeWidth="1"/>
                    <circle cx="60" cy="45" r="15" fill="none" stroke="#ff9800" strokeWidth="1.5" strokeDasharray="3,2"/>
                    <circle cx="60" cy="42" r="6" fill="#ff9800" opacity="0.4"/>
                    <ellipse cx="60" cy="55" rx="10" ry="6" fill="#ff9800" opacity="0.3"/>
                    <rect x="35" y="25" width="50" height="50" rx="4" fill="none" stroke="#ff9800" strokeWidth="2"/>
                    <text x="60" y="112" textAnchor="middle" fill="#888" fontSize="11">Frame + Detection</text>
                    <path d="M125 55 L165 55" stroke="#ff9800" strokeWidth="2" markerEnd="url(#arrow2)"/>
                    <defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ff9800"/></marker></defs>
                    <rect x="175" y="20" width="70" height="70" rx="8" fill="#1a1a2e" stroke="#ff9800" strokeWidth="2"/>
                    <circle cx="210" cy="45" r="12" fill="none" stroke="#ff9800" strokeWidth="1.5"/>
                    <circle cx="210" cy="42" r="5" fill="#ff9800" opacity="0.5"/>
                    <ellipse cx="210" cy="52" rx="8" ry="5" fill="#ff9800" opacity="0.4"/>
                    <text x="210" y="112" textAnchor="middle" fill="#888" fontSize="11">Cropped Face (+30% margin)</text>
                    <text x="340" y="40" fill="#ff9800" fontSize="11" fontWeight="600">{'\u2713'} 95%+ confidence</text>
                    <text x="340" y="58" fill="#888" fontSize="10">30% margin padding</text>
                    <text x="340" y="76" fill="#888" fontSize="10">Context preserved</text>
                </svg>
            ),
        },
        {
            number: 3,
            title: 'Dataset Preparation',
            color: '#4caf50',
            description: 'Cropped face images are organized into "real" and "fake" categories based on FaceForensics++ metadata. Small or corrupted images (<90px) are filtered out. The dataset is then split into training (80%), validation (10%), and test (10%) sets using stratified splitting.',
            details: [
                'Labels faces as REAL or FAKE using FaceForensics++ CSV metadata',
                'Filters out low-quality images smaller than 90\u00d790 pixels',
                'Balances fake samples across manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter)',
                'Splits into train/val/test (80/10/10) with stratified random sampling',
            ],
            diagram: (
                <svg viewBox="0 0 400 120" className="step-illustration">
                    <g>
                        <rect x="10" y="15" width="55" height="40" rx="4" fill="rgba(76,175,80,0.15)" stroke="#4caf50" strokeWidth="1.5"/>
                        <text x="37" y="39" textAnchor="middle" fill="#4caf50" fontSize="11" fontWeight="600">REAL</text>
                        <rect x="10" y="65" width="55" height="40" rx="4" fill="rgba(244,67,54,0.15)" stroke="#f44336" strokeWidth="1.5"/>
                        <text x="37" y="89" textAnchor="middle" fill="#f44336" fontSize="11" fontWeight="600">FAKE</text>
                    </g>
                    <path d="M80 55 L120 55" stroke="#4caf50" strokeWidth="2" markerEnd="url(#arrow3)"/>
                    <defs><marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4caf50"/></marker></defs>
                    <rect x="130" y="8" width="100" height="28" rx="4" fill="rgba(76,175,80,0.1)" stroke="#4caf50" strokeWidth="1"/>
                    <text x="180" y="26" textAnchor="middle" fill="#4caf50" fontSize="10">Train 80%</text>
                    <rect x="130" y="44" width="100" height="28" rx="4" fill="rgba(255,152,0,0.1)" stroke="#ff9800" strokeWidth="1"/>
                    <text x="180" y="62" textAnchor="middle" fill="#ff9800" fontSize="10">Val 10%</text>
                    <rect x="130" y="80" width="100" height="28" rx="4" fill="rgba(108,140,255,0.1)" stroke="#6c8cff" strokeWidth="1"/>
                    <text x="180" y="98" textAnchor="middle" fill="#6c8cff" fontSize="10">Test 10%</text>
                    <text x="310" y="30" fill="#888" fontSize="10">{'\u2713'} Min 90\u00d790px filter</text>
                    <text x="310" y="50" fill="#888" fontSize="10">{'\u2713'} Stratified split</text>
                    <text x="310" y="70" fill="#888" fontSize="10">{'\u2713'} Multi-method balance</text>
                    <text x="310" y="90" fill="#888" fontSize="10">{'\u2713'} CSV metadata labels</text>
                </svg>
            ),
        },
        {
            number: 4,
            title: 'CNN Training (EfficientNetB0)',
            color: '#f44336',
            description: 'A two-phase transfer learning approach trains an EfficientNetB0-based classifier. Phase 1 freezes the pre-trained ImageNet backbone and trains only the classification head. Phase 2 unfreezes the entire network for fine-tuning with a very low learning rate, achieving ~92% accuracy.',
            details: [
                'EfficientNetB0 backbone pre-trained on ImageNet (224\u00d7224 input)',
                'Phase 1: Frozen base, train head only (lr=1e-3, up to 15 epochs)',
                'Phase 2: Full fine-tuning (lr=1e-5, up to 30 epochs)',
                'Data augmentation: rotation, flip, zoom, shift, brightness',
                'Class weight balancing for imbalanced datasets',
                'Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint',
                'Output: binary sigmoid \u2014 score > 0.5 = REAL, \u2264 0.5 = FAKE',
            ],
            diagram: (
                <svg viewBox="0 0 400 140" className="step-illustration">
                    <rect x="10" y="30" width="70" height="80" rx="6" fill="#1a1a2e" stroke="#f44336" strokeWidth="1.5"/>
                    <text x="45" y="55" textAnchor="middle" fill="#f44336" fontSize="9" fontWeight="600">224\u00d7224</text>
                    <text x="45" y="70" textAnchor="middle" fill="#888" fontSize="8">Face Input</text>
                    <text x="45" y="95" textAnchor="middle" fill="#666" fontSize="8">preprocess</text>
                    <text x="45" y="106" textAnchor="middle" fill="#666" fontSize="8">[-1, 1]</text>
                    <path d="M90 70 L115 70" stroke="#f44336" strokeWidth="1.5" markerEnd="url(#arrow4)"/>
                    <rect x="120" y="15" width="100" height="110" rx="6" fill="#1a1a2e" stroke="#ff9800" strokeWidth="1.5"/>
                    <text x="170" y="35" textAnchor="middle" fill="#ff9800" fontSize="10" fontWeight="600">EfficientNetB0</text>
                    <text x="170" y="52" textAnchor="middle" fill="#888" fontSize="8">(ImageNet weights)</text>
                    {[0,1,2,3].map(i => (
                        <rect key={i} x="135" y={60 + i*14} width="70" height="10" rx="2" fill={`rgba(255,152,0,${0.15 + i*0.1})`} stroke="#ff9800" strokeWidth="0.5"/>
                    ))}
                    <defs><marker id="arrow4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#f44336"/></marker></defs>
                    <path d="M230 70 L255 70" stroke="#f44336" strokeWidth="1.5" markerEnd="url(#arrow4)"/>
                    <rect x="260" y="25" width="80" height="90" rx="6" fill="#1a1a2e" stroke="#6c8cff" strokeWidth="1.5"/>
                    <text x="300" y="45" textAnchor="middle" fill="#6c8cff" fontSize="9" fontWeight="600">Head</text>
                    <text x="300" y="62" textAnchor="middle" fill="#888" fontSize="7">GlobalAvgPool</text>
                    <text x="300" y="74" textAnchor="middle" fill="#888" fontSize="7">BatchNorm</text>
                    <text x="300" y="86" textAnchor="middle" fill="#888" fontSize="7">Dense(256)+Dropout</text>
                    <text x="300" y="98" textAnchor="middle" fill="#888" fontSize="7">Dense(1, sigmoid)</text>
                    <path d="M350 70 L375 70" stroke="#4caf50" strokeWidth="1.5" markerEnd="url(#arrow5)"/>
                    <defs><marker id="arrow5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4caf50"/></marker></defs>
                    <text x="388" y="65" fill="#4caf50" fontSize="11" fontWeight="700">0\u20131</text>
                    <text x="388" y="80" fill="#888" fontSize="8">Score</text>
                </svg>
            ),
        },
    ];

    return (
        <section className="tech-page">
            <div className="tech-hero">
                <h1 className="tech-title">How It Works</h1>
                <p className="tech-subtitle">
                    Our deepfake detection pipeline processes videos through four stages — from raw video
                    to a trained AI model that scores each face for authenticity.
                </p>
            </div>

            {/* Pipeline overview */}
            <div className="pipeline-overview">
                {steps.map((step, i) => (
                    <React.Fragment key={step.number}>
                        <div className="pipeline-step-mini">
                            <StepDiagram number={step.number} title={step.title} color={step.color} />
                            <span style={{ color: step.color, fontWeight: 600, fontSize: 13 }}>{step.title}</span>
                        </div>
                        {i < steps.length - 1 && <FlowArrow />}
                    </React.Fragment>
                ))}
            </div>

            {/* Detailed steps */}
            {steps.map((step) => (
                <div className="tech-step" key={step.number}>
                    <div className="tech-step-header">
                        <StepDiagram number={step.number} title={step.title} color={step.color} />
                        <div>
                            <h2 className="tech-step-title" style={{ color: step.color }}>
                                Step {step.number}: {step.title}
                            </h2>
                        </div>
                    </div>
                    <p className="tech-step-desc">{step.description}</p>
                    <div className="tech-step-diagram">
                        {step.diagram}
                    </div>
                    <ul className="tech-step-details">
                        {step.details.map((d, i) => <li key={i}>{d}</li>)}
                    </ul>
                </div>
            ))}

            {/* Inference section */}
            <div className="tech-step">
                <div className="tech-step-header">
                    <svg width="80" height="80" viewBox="0 0 80 80" className="step-icon">
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#6c8cff" strokeWidth="2.5"/>
                        <text x="40" y="46" textAnchor="middle" fill="#6c8cff" fontSize="22" fontWeight="700">{'\u25b6'}</text>
                    </svg>
                    <div>
                        <h2 className="tech-step-title" style={{ color: '#6c8cff' }}>
                            Real-Time Inference
                        </h2>
                    </div>
                </div>
                <p className="tech-step-desc">
                    When you upload a video, the app uses YOLOv8 for fast face detection on each frame,
                    then feeds cropped faces through the trained EfficientNetB0 model. Each face gets an
                    authenticity score (0 = fake, 1 = real), and the processed video shows bounding boxes
                    with per-face REAL/FAKE labels overlaid in real time.
                </p>
                <div className="tech-step-diagram">
                    <svg viewBox="0 0 400 80" className="step-illustration">
                        <rect x="5" y="15" width="65" height="50" rx="6" fill="#1a1a2e" stroke="#6c8cff" strokeWidth="1.5"/>
                        <text x="37" y="44" textAnchor="middle" fill="#6c8cff" fontSize="9" fontWeight="600">Upload</text>
                        <path d="M80 40 L105 40" stroke="#6c8cff" strokeWidth="1.5" markerEnd="url(#arrowI)"/>
                        <rect x="110" y="15" width="65" height="50" rx="6" fill="#1a1a2e" stroke="#ff9800" strokeWidth="1.5"/>
                        <text x="142" y="38" textAnchor="middle" fill="#ff9800" fontSize="9" fontWeight="600">YOLOv8</text>
                        <text x="142" y="50" textAnchor="middle" fill="#888" fontSize="7">Face Detect</text>
                        <path d="M185 40 L210 40" stroke="#ff9800" strokeWidth="1.5" markerEnd="url(#arrowI)"/>
                        <rect x="215" y="15" width="65" height="50" rx="6" fill="#1a1a2e" stroke="#f44336" strokeWidth="1.5"/>
                        <text x="247" y="38" textAnchor="middle" fill="#f44336" fontSize="8" fontWeight="600">EfficientNet</text>
                        <text x="247" y="50" textAnchor="middle" fill="#888" fontSize="7">Predict</text>
                        <path d="M290 40 L315 40" stroke="#4caf50" strokeWidth="1.5" markerEnd="url(#arrowI)"/>
                        <rect x="320" y="15" width="70" height="50" rx="6" fill="#1a1a2e" stroke="#4caf50" strokeWidth="1.5"/>
                        <text x="355" y="38" textAnchor="middle" fill="#4caf50" fontSize="9" fontWeight="600">REAL</text>
                        <text x="355" y="50" textAnchor="middle" fill="#f44336" fontSize="9" fontWeight="600">FAKE</text>
                        <defs><marker id="arrowI" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6c8cff"/></marker></defs>
                    </svg>
                </div>
            </div>
        </section>
    );
}
