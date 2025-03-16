import React, { useState } from 'react';
import { FileInput, Button, Spinner, Progress } from 'flowbite-react';
import { NavBar } from '../components/NavBar';
import { FooterComponent } from '../components/FooterComponent';
import '../styles/Footer.css';

export default function Sentinel() {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // Mapping for predicted class values
  const classMap = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake',
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadUrl = 'http://127.0.0.1:5000/sentinel'; // Change this to your backend API URL

      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
        headers: {
          // Add any headers here if needed, such as authorization tokens
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json(); // Assuming response returns the classification result
      setPredictionResult(data); // Store the prediction result in state

      setIsUploading(false);
      alert('File uploaded and processed successfully!');
    } catch (error) {
      setIsUploading(false);
      setError(error.message);
    }
  };

  // Function to handle downloading the classification results
  const handleDownload = () => {
    const dataToDownload = {
      predicted_class: classMap[predictionResult.predicted_class],
      probability: predictionResult.probability,
    };

    const blob = new Blob([JSON.stringify(dataToDownload, null, 2)], {
      type: 'application/json',
    });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'classification_result.json';
    link.click();
  };

  return (
    <>
      <NavBar />
      <div className="max-w-md mx-auto mt-10">
        <h2 className="text-2xl font-bold mb-4">Sentinel - File Upload</h2>

        <FileInput
          id="large-file-upload"
          className="mb-4"
          helperText="Please upload your dataset (e.g., images, zip files)"
          onChange={handleFileChange}
        />

        {error && <p className="text-red-500">{error}</p>}

        {isUploading && (
          <div className="mb-4">
            <Spinner aria-label="File uploading..." />
            <p>Uploading... {uploadProgress}%</p>
            <Progress progress={uploadProgress} />
          </div>
        )}

        <Button onClick={handleUpload} disabled={isUploading || !file}>
          {isUploading ? 'Uploading...' : 'Upload File'}
        </Button>

        {predictionResult && (
          <div className="mt-4">
            <h3 className="text-lg font-semibold">Classification Result</h3>
            <p>Predicted Class: {classMap[predictionResult.predicted_class]}</p>
            <div className="mt-4">
              <h4 className="font-semibold">Probabilities</h4>
              <table className="min-w-full table-auto">
                <thead>
                  <tr>
                    <th className="px-4 py-2">Class</th>
                    <th className="px-4 py-2">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(classMap).map((key) => (
                    <tr key={key}>
                      <td className="border px-4 py-2">{classMap[key]}</td>
                      <td className="border px-4 py-2">
                        {predictionResult.probability[key] || 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Button gradientDuoTone="tealToLime" onClick={handleDownload}>
            Download Classification Results
          </Button>
        </div>
      </div>
    </>
  );
}
