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
        setError('No processed images or data available for download.');
        return;
    }

    const zip = new JSZip();
    console.log({jsons})
    // Fetch and add GeoJSON files to ZIP
    const geojsonPromises = jsons.map(async (data, index) => {
      console.log("Reached here")
        if (data) {
            const filename = data.split('/').pop();
            console.log(`Fetching GeoJSON: ${filename}`);

            try {
                const response = await fetch(`http://localhost:5000/download_geojson/${filename}`);

                if (!response.ok) {
                    throw new Error(`Failed to fetch ${filename}: ${response.statusText}`);
                }

                console.log(`Successfully fetched: ${filename}`);

                const geojsonBlob = await response.blob();

                // Use a consistent naming pattern and ensure files[index] exists before using it
                const geojsonFileName = files[index] ? `detection_${files[index].name}.geojson` : `detection_${index}.geojson`;
                zip.file(geojsonFileName, geojsonBlob);
            } catch (error) {
                console.error(`GeoJSON download error for ${filename}:`, error);
            }
        } else {
            console.warn(`Missing json_path for detection index ${index}`);
        }
    });

    // Add processed images to ZIP
    images.forEach((image, index) => {
        const base64Data = image.split(',')[1];
        zip.file(`processed_${files[index].name}`, base64Data, { base64: true });
    });

    // Wait for all GeoJSON files to be added before generating the ZIP
    await Promise.all(geojsonPromises);

    // Generate and trigger download of the ZIP file
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    saveAs(zipBlob, 'marine_debris_results.zip');
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
                      <th className="border border-gray-300 px-4 py-2">Top-Left X</th>
                      <th className="border border-gray-300 px-4 py-2">Top-Left Y</th>
                      <th className="border border-gray-300 px-4 py-2">Bottom-Right X</th>
                      <th className="border border-gray-300 px-4 py-2">Bottom-Right Y</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detections[index].map((box, i) => (
                      <tr key={i} className="text-center">
                        <td className="border border-gray-300 px-4 py-2">{i + 1}</td>
                        <td className="border border-gray-300 px-4 py-2">{box[0]}</td>
                        <td className="border border-gray-300 px-4 py-2">{box[1]}</td>
                        <td className="border border-gray-300 px-4 py-2">{box[2]}</td>
                        <td className="border border-gray-300 px-4 py-2">{box[3]}</td>
                      </tr>
                    ))}
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
