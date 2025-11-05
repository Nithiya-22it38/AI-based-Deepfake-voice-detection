// import React, { useState } from "react";
// import { ReactMic } from "react-mic";
// import { Button, Card, CardContent, Typography, LinearProgress } from "@mui/material";
// import MicIcon from "@mui/icons-material/Mic";
// import StopIcon from "@mui/icons-material/Stop";

// export default function VoiceAnalyzer() {
//   const [record, setRecord] = useState(false);
//   const [result, setResult] = useState(null);
//   const [confidence, setConfidence] = useState(null);
//   const [reason, setReason] = useState("");

//   const startRecording = () => {
//     setRecord(true);
//     setResult(null);
//   };

//   const stopRecording = () => {
//     setRecord(false);
//     setTimeout(() => analyzeAudio(), 1500);
//   };

//   const analyzeAudio = async () => {
//     // 🧠 Simulate backend model output for now
//     const simulated = Math.random() > 0.5;
//     const conf = (Math.random() * (0.98 - 0.8) + 0.8).toFixed(2);
//     const reasonText = simulated
//       ? "Pitch stability and spectral centroid match genuine human voice patterns."
//       : "Detected irregular MFCC variation typical of synthetic audio.";

//     setResult(simulated ? "REAL" : "FAKE");
//     setConfidence(conf);
//     setReason(reasonText);
//   };

//   return (
//     <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: 80 }}>
//       <Typography variant="h4" gutterBottom>
//         🎙 Deepfake Voice Detector
//       </Typography>

//       <ReactMic
//         record={record}
//         className="sound-wave"
//         strokeColor="#4caf50"
//         backgroundColor="#f5f5f5"
//       />

//       <Button
//         variant="contained"
//         color={record ? "error" : "primary"}
//         startIcon={record ? <StopIcon /> : <MicIcon />}
//         onClick={record ? stopRecording : startRecording}
//         sx={{ marginTop: 3, borderRadius: 3, paddingX: 4 }}
//       >
//         {record ? "Stop Recording" : "Start Recording"}
//       </Button>

//       {result && (
//         <Card sx={{ marginTop: 4, width: "80%", maxWidth: 500, boxShadow: 4, borderRadius: 3 }}>
//           <CardContent>
//             <Typography variant="h6">
//               Result: <b style={{ color: result === "REAL" ? "green" : "red" }}>{result}</b>
//             </Typography>
//             <Typography variant="body1" sx={{ marginTop: 1 }}>
//               Confidence: {(confidence * 100).toFixed(1)}%
//             </Typography>
//             <LinearProgress
//               variant="determinate"
//               value={confidence * 100}
//               sx={{ height: 8, borderRadius: 5, marginTop: 1 }}
//             />
//             <Typography variant="body2" sx={{ marginTop: 2, color: "gray" }}>
//               {reason}
//             </Typography>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   );
// }

import React, { useState } from "react";
import { ReactMic } from "react-mic";
import axios from "axios";
import MicIcon from "@mui/icons-material/Mic";
import StopIcon from "@mui/icons-material/Stop";
import { Button, Card, Typography, CircularProgress } from "@mui/material";

const VoiceAnalyzer = () => {
  const [record, setRecord] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const startRecording = () => setRecord(true);
  const stopRecording = () => setRecord(false);

  const onStop = async (recordedBlob) => {
    console.log("Recording stopped: ", recordedBlob);
    setLoading(true);
    setResult(null);

    // Prepare form data
    const formData = new FormData();
    formData.append("audio", recordedBlob.blob, "sample.wav");

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Backend not reachable or model failed." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <Card className="p-8 rounded-2xl shadow-xl text-center bg-white w-96">
        <Typography variant="h5" className="mb-4 font-bold text-gray-700">
          🎤 Deepfake Voice Detector
        </Typography>

        <ReactMic
          record={record}
          className="w-full mb-4"
          onStop={onStop}
          strokeColor="#3b82f6"
          backgroundColor="#e0e7ff"
        />

        {!record ? (
          <Button
            variant="contained"
            color="primary"
            startIcon={<MicIcon />}
            onClick={startRecording}
          >
            Start Recording
          </Button>
        ) : (
          <Button
            variant="contained"
            color="error"
            startIcon={<StopIcon />}
            onClick={stopRecording}
          >
            Stop Recording
          </Button>
        )}

        {loading && (
          <div className="mt-6">
            <CircularProgress />
            <Typography variant="body2" className="mt-2 text-gray-600">
              Analyzing your voice...
            </Typography>
          </div>
        )}

        {result && (
          <div className="mt-6">
            {result.error ? (
              <Typography color="error">{result.error}</Typography>
            ) : (
              <>
                <Typography variant="h6" className="text-gray-800">
                  Prediction: <b>{result.prediction}</b>
                </Typography>
                <Typography variant="body2" className="text-gray-600 mt-2">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </Typography>
                <Typography variant="body2" className="text-gray-500 mt-1 italic">
                  Reason: {result.reason || "Model interpretation unavailable."}
                </Typography>
              </>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default VoiceAnalyzer;
