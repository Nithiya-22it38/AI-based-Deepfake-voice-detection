import React, { useState } from "react";
import { ReactMic } from "react-mic";

function MicRecorder() {
  const [record, setRecord] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [result, setResult] = useState("");

  const startRecording = () => {
    setRecord(true);
  };

  const stopRecording = () => {
    setRecord(false);
  };

  const onStop = (recordedBlob) => {
    console.log("Recorded blob:", recordedBlob);
    setAudioBlob(recordedBlob.blob);
  };

  const sendToBackend = async () => {
    if (!audioBlob) return alert("Please record your voice first!");

    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data.result); // backend will return “Real” or “Fake”
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>🎤 Deepfake Voice Detector</h2>

      <ReactMic
        record={record}
        className="sound-wave"
        onStop={onStop}
        strokeColor="#000000"
        backgroundColor="#FFB6C1"
      />

      <div style={{ marginTop: "20px" }}>
        <button onClick={startRecording}>Start 🎙️</button>
        <button onClick={stopRecording}>Stop ⏹️</button>
        <button onClick={sendToBackend}>Check Authenticity 🔍</button>
      </div>

      {result && (
        <h3 style={{ marginTop: "30px" }}>
          Result: {result === "Real" ? "✅ Real Voice" : "❌ Fake Voice"}
        </h3>
      )}
    </div>
  );
}

export default MicRecorder;
