// MarineDebris.js
import React, { useState } from 'react';
import { FileInput, Button, Alert } from 'flowbite-react';
import { NavBar } from '../components/NavBar';
import axios from 'axios';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

export default function ObjectDetection() {
  const [files, setFiles] = useState([]);
  const [images, setImages] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [jsons, setJsons] = useState([]);

  const handleFileChange = (e) => {
    setFiles([...e.target.files]);
    setImages([]);
    setDetections([]);
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one image file to upload');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);
    setImages([]);
    setDetections([]);
    setJsons([]);

    const uploadedImages = [];
    const detectedObjects = [];
    const uploadedJsons = [];

    for (let i = 0; i < files.length; i++) {
      const formData = new FormData();
      formData.append('file', files[i]);

      try {
        const response = await axios.post('http://localhost:5000/marinedebris/detect', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const progress = ((progressEvent.loaded / progressEvent.total) * 100) / files.length;
            setUploadProgress((prev) => prev + progress);
          },
        });

        uploadedImages.push(`data:image/jpeg;base64,${response.data.image_base64}`);
        detectedObjects.push(response.data.detections);
        uploadedJsons.push(response.data.json_path);
      } catch (error) {
        setError(`Upload failed for file ${files[i].name}. Please try again.`);
      }
    }

    setImages(uploadedImages);
    setDetections(detectedObjects);
    setJsons(uploadedJsons);
    setIsUploading(false);
  };

  const handleDownload = async () => {
    if (images.length === 0 || detections.length === 0) {
      setError('No processed images or data available for download');
      return;
    }
  
    setIsDownloading(true);
    setError(null);
  
    try {
      const zip = new JSZip();
      
      // Process each file in parallel
      await Promise.all(files.map(async (file, index) => {
        const baseName = file.name.split('.')[0];
        
        // 1. Add GeoJSON
        try {
          const geojsonFilename = `${baseName}.geojson`;
          const geojsonResponse = await fetch(
            `http://localhost:5000/download_geojson/${geojsonFilename}`
          );
          if (geojsonResponse.ok) {
            zip.file(`${baseName}.geojson`, await geojsonResponse.blob());
          }
        } catch (error) {
          console.error(`GeoJSON error for ${file.name}:`, error);
        }
  
        // 2. Add Processed Image
        try {
          const processedResponse = await fetch(
            `http://localhost:5000/download_planetscope_processed/${file.name}`
          );
          if (processedResponse.ok) {
            zip.file(`${baseName}_processed.jpg`, await processedResponse.blob());
          }
        } catch (error) {
          console.error(`Processed image error for ${file.name}:`, error);
        }
  
        // 3. Add Original Image (optional)
        try {
          const originalResponse = await fetch(
            `http://localhost:5000/download_planetscope_original/${file.name}`
          );
          if (originalResponse.ok) {
            zip.file(`${baseName}_original.jpg`, await originalResponse.blob());
          }
        } catch (error) {
          console.error(`Original image error for ${file.name}:`, error);
        }
  
        // 4. Add Base64 Preview
        if (images[index]) {
          const base64Data = images[index].split(',')[1];
          zip.file(`${baseName}_preview.jpg`, base64Data, { base64: true });
        }
      }));
  
      // Generate and download zip
      const content = await zip.generateAsync({ type: 'blob' });
      saveAs(content, 'planetscope_marine_debris_results.zip');
  
    } catch (error) {
      setError('Failed to prepare download package');
      console.error('Download error:', error);
    } finally {
      setIsDownloading(false);
    }
  };


  return (
    <>
      <NavBar />
      <div className="max-w-md mx-auto mt-10">
        <h2 className="text-2xl font-bold mb-4">Marine Debris Object Detection - File Upload</h2>

        <FileInput
          id="file-upload"
          className="mb-4"
          helperText="Upload multiple images for marine debris detection."
          onChange={handleFileChange}
          multiple
        />

        {error && <Alert color="failure">{error}</Alert>}

        {isUploading && (
          <div className="mb-4">
            <p>Uploading... {uploadProgress.toFixed(0)}%</p>
          </div>
        )}

        <Button onClick={handleUpload} disabled={isUploading || files.length === 0}>
          {isUploading ? 'Uploading...' : 'Upload Images'}
        </Button>

        {images.map((src, index) => (
          <div key={index} className="mt-4">
            <img src={src} alt={`Processed Detection ${index + 1}`} className="max-w-full rounded-md" />
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Detected Marine Debris ({files[index].name}):</h3>
              {detections[index]?.length > 0 ? (
                <table className="min-w-full border-collapse border border-gray-300 mt-2">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-4 py-2">#</th>
                      <th className="border border-gray-300 px-4 py-2">ID</th>
                      <th className="border border-gray-300 px-4 py-2">AREA (M²)</th>
                      <th className="border border-gray-300 px-4 py-2">Top-Left X</th>
                      <th className="border border-gray-300 px-4 py-2">Top-Left Y</th>
                      <th className="border border-gray-300 px-4 py-2">Bottom-Right X</th>
                      <th className="border border-gray-300 px-4 py-2">Bottom-Right Y</th>
                    </tr>
                  </thead>
                  <tbody>
            {detections[index].map((box, i) => {
              // Ensure box has enough elements before destructuring
              const [id, area, x0, y0, x1, y1] = box.length >= 6 ? box : [null, null, null, null, null, null];

              return (
                <tr key={i} className="text-center">
                  <td className="border border-gray-300 px-4 py-2">{i + 1}</td>
                  <td className="border border-gray-300 px-4 py-2">{id ?? "N/A"}</td>
                  <td className="border border-gray-300 px-4 py-2">{area ? area.toFixed(2) : "N/A"}</td>
                  <td className="border border-gray-300 px-4 py-2">{x0 ?? "N/A"}</td>
                  <td className="border border-gray-300 px-4 py-2">{y0 ?? "N/A"}</td>
                  <td className="border border-gray-300 px-4 py-2">{x1 ?? "N/A"}</td>
                  <td className="border border-gray-300 px-4 py-2">{y1 ?? "N/A"}</td>
                </tr>
              );
            })}
          </tbody>
                </table>
              ) : (
                <p>No marine debris detected.</p>
              )}
            </div>
            
          </div>
        ))}
        {images.length > 0 && (
          <Button onClick={handleDownload} className="mt-4">
            Download All Results (.zip)
          </Button>
        )}
      </div>
    </>
  );
}
