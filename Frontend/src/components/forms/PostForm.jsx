import React, { useState, useRef, useEffect } from "react";

/* ─── Client-side Image Compression ─── */
const compressImage = (file, maxWidth = 1000, maxHeight = 1000, quality = 0.7) => {
  return new Promise((resolve) => {
    if (!file.type.startsWith("image/")) {
      resolve(null);
      return;
    }
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = () => {
        const canvas = document.createElement("canvas");
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }
        }
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL("image/jpeg", quality));
      };
      img.onerror = () => resolve(null);
    };
    reader.onerror = () => resolve(null);
  });
};


/* ─── Result Dialog ──────────────────────────────────────────────────── */
const ResultDialog = ({ result, predictionTime, onSave, onDiscard }) => {
  const isFake = result === "fake";
  const accentColor = isFake ? "#ef4444" : "#22c55e";
  const bgColor     = isFake ? "#1a0505" : "#051a0a";
  const borderColor = isFake ? "#ef444466" : "#22c55e66";

  return (
    <div
      style={{
        position: "fixed", inset: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.80)",
        backdropFilter: "blur(6px)",
        zIndex: 9999,
      }}
    >
      <div
        style={{
          background: bgColor,
          border: `2px solid ${accentColor}`,
          borderRadius: 20,
          padding: "40px 36px",
          maxWidth: 500,
          width: "92%",
          textAlign: "center",
          boxShadow: `0 0 60px ${borderColor}`,
          animation: "resultFadeIn 0.3s ease",
        }}
      >
        <style>{`
          @keyframes resultFadeIn {
            from { opacity: 0; transform: scale(0.90) translateY(20px); }
            to   { opacity: 1; transform: scale(1)   translateY(0);     }
          }
        `}</style>

        {/* Icon */}
        <div style={{ fontSize: 60, marginBottom: 16 }}>
          {isFake ? "🚨" : "✅"}
        </div>

        {/* Title */}
        <h2
          style={{
            color: accentColor,
            fontSize: 22,
            fontWeight: 900,
            letterSpacing: "0.5px",
            marginBottom: 16,
            textTransform: "uppercase",
          }}
        >
          {isFake ? "Deepfake Detected" : "Authentic Content Detected"}
        </h2>

        {/* Description */}
        <p
          style={{
            color: "#d1d5db",
            fontSize: 15,
            lineHeight: 1.7,
            marginBottom: 8,
          }}
        >
          {isFake
            ? "The uploaded file shows characteristics of a deepfake and may be manipulated or artificially generated."
            : "The uploaded file appears to be genuine with no significant signs of manipulation."}
        </p>
        <p
          style={{
            color: isFake ? "#fca5a5" : "#86efac",
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 24,
          }}
        >
          {isFake
            ? "Please verify its authenticity before using or sharing."
            : "You can proceed with normal usage."}
        </p>

        {/* Prediction time */}
        {predictionTime != null && (
          <p style={{ color: "#6b7280", fontSize: 12, marginBottom: 24 }}>
            ⏱ Prediction time:{" "}
            <strong style={{ color: "#9ca3af" }}>{predictionTime}s</strong>
          </p>
        )}

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <button
            onClick={onSave}
            style={{
              background: accentColor,
              color: "#fff",
              border: "none",
              borderRadius: 10,
              padding: "12px 28px",
              fontWeight: 700,
              fontSize: 15,
              cursor: "pointer",
              transition: "opacity 0.2s",
              minWidth: 140,
            }}
            onMouseOver={(e) => (e.currentTarget.style.opacity = 0.85)}
            onMouseOut={(e)  => (e.currentTarget.style.opacity = 1)}
          >
            {isFake ? "Save Anyway" : "Save to Home"}
          </button>

          <button
            onClick={onDiscard}
            style={{
              background: "transparent",
              color: "#9ca3af",
              border: "1px solid #374151",
              borderRadius: 10,
              padding: "12px 28px",
              fontWeight: 600,
              fontSize: 15,
              cursor: "pointer",
              transition: "all 0.2s",
              minWidth: 140,
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.color = "#fff";
              e.currentTarget.style.borderColor = "#6b7280";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.color = "#9ca3af";
              e.currentTarget.style.borderColor = "#374151";
            }}
          >
            Don't Save
          </button>
        </div>
      </div>
    </div>
  );
};

/* ─── PostForm ──────────────────────────────────────────────────────── */
const PostForm = ({ formData, handleChange, onSubmit }) => {
  const [mediaPreview, setMediaPreview]         = useState(null);
  const [title, setTitle]                       = useState("");
  const [description, setDescription]           = useState("");
  const [isAnalyzing, setIsAnalyzing]           = useState(false);
  const [predictionResult, setPredictionResult] = useState(null); // null | "real" | "fake"
  const [predictionTime, setPredictionTime]     = useState(null);
  const fileInputRef                            = useRef(null);
  const pendingPayload                          = useRef(null);

  const handleMediaChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        const compressed = await compressImage(file);
        setMediaPreview(compressed);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          setMediaPreview(reader.result);
        };
        reader.readAsDataURL(file);
      }
      handleChange(e);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const file = e.target.media.files[0];
    if (!file) { alert("Please select a file first."); return; }

    const fd = new FormData();
    const fileType = file.type.split("/")[0];
    fd.append(fileType, file);

    setPredictionResult(null);
    setIsAnalyzing(true);

    try {
      const response = await fetch("http://127.0.0.1:5001/upload", {
        method: "POST",
        body: fd,
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const data = await response.json();
      const label     = data[1];
      const timeTaken = data[0]?.prediction_time_s ?? null;

      // Store what we'd save, but wait for user confirmation
      pendingPayload.current = { mediaPreview, title, description, predictionLabel: label };

      setIsAnalyzing(false);
      setPredictionResult(label);
      setPredictionTime(timeTaken);

    } catch (err) {
      console.error("Prediction error:", err);
      setIsAnalyzing(false);
      alert("Error contacting prediction server. Please ensure the backend is running on port 5001.");
    }
  };

  /* ── User clicks Save ── */
  const handleSave = () => {
    try {
      if (pendingPayload.current) {
        onSubmit(pendingPayload.current);
      }
      resetForm();
      // Optional: Add a success notification or brief delay if needed
    } catch (err) {
      console.error("Save error:", err);
      alert("Failed to save post. Your storage might be full.");
    }
  };

  /* ── User clicks Don't Save ── */
  const handleDiscard = () => {
    resetForm();
  };

  const resetForm = () => {
    setPredictionResult(null);
    setPredictionTime(null);
    setMediaPreview(null);
    setTitle("");
    setDescription("");
    pendingPayload.current = null;
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <>
      {/* Result dialog */}
      {predictionResult && (
        <ResultDialog
          result={predictionResult}
          predictionTime={predictionTime}
          onSave={handleSave}
          onDiscard={handleDiscard}
        />
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* Title */}
        <div>
          <label htmlFor="title" className="text-white font-semibold mb-1">
            Title
          </label>
          <input
            type="text"
            name="title"
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter post title"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white shad-input"
            style={{ width: "100%", height: "40px" }}
          />
        </div>

        {/* Description */}
        <div>
          <label htmlFor="description" className="text-white font-semibold mb-1">
            Description
          </label>
          <textarea
            name="description"
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Enter post description"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white resize-none shad-input"
            style={{ width: "100%", height: "100px" }}
            rows="4"
          />
        </div>

        {/* Media upload */}
        <div>
          <label htmlFor="media" className="text-white font-semibold mb-1">
            Media (Image, Audio, or Video)
          </label>
          <input
            ref={fileInputRef}
            type="file"
            name="media"
            id="media"
            onChange={handleMediaChange}
            accept="image/*, audio/*, video/*"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white shad-input"
            style={{ width: "100%", height: "40px" }}
          />
          {mediaPreview && (
            <div className="mt-2">
              {mediaPreview.startsWith("data:image/") && (
                <img
                  src={mediaPreview}
                  alt="Preview"
                  className="rounded"
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    objectFit: "contain",
                    backgroundColor: "#101010",
                  }}
                />
              )}
              {mediaPreview.startsWith("data:audio/") && (
                <audio controls>
                  <source src={mediaPreview} />
                </audio>
              )}
              {mediaPreview.startsWith("data:video/") && (
                <video controls style={{ maxWidth: "100%" }}>
                  <source src={mediaPreview} />
                </video>
              )}
            </div>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={isAnalyzing}
          className="text-white px-4 py-2 rounded-md transition-colors duration-300 hover:opacity-85"
          style={{
            backgroundColor: "#877EFF",
            opacity: isAnalyzing ? 0.5 : 1,
            cursor: isAnalyzing ? "not-allowed" : "pointer",
          }}
        >
          {isAnalyzing ? "Analyzing..." : "Create Post"}
        </button>
      </form>
    </>
  );
};

export default PostForm;
