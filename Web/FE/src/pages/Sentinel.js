import React, { useState } from 'react';
import { FileInput, Button, Alert } from 'flowbite-react';
import { NavBar } from '../components/NavBar';
import axios from 'axios';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';


export default function SentinelUpload() {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [detectedImage, setDetectedImage] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [originalTiffUrl, setOriginalTiffUrl] = useState(null);
  const [geojsonPath, setGeojsonPath] = useState(null);

  const handleFileChange = (e) => {
    setFiles([...e.target.files]);
    setDetections([]);
    setDetectedImage(null);
    setOriginalTiffUrl(URL.createObjectURL(e.target.files[0]));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select a .tif file to upload');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', files[0]);

    try {
      const response = await axios.post('http://127.0.0.1:5000/sentinel', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          setUploadProgress(progress);
        },
      });

      const { bboxes, image, geojson_path } = response.data;
      setDetections([bboxes]);
      setDetectedImage(`data:image/jpeg;base64,${image}`);
      setGeojsonPath(geojson_path);
    } catch (error) {
      console.error("Upload error:", error.response?.data || error.message);
      setError(error.response?.data?.error || 'Upload failed. Please try again.');
    }

    setIsUploading(false);
  };

  const handleDownload = async () => {
    if (!files.length || !detectedImage) {
      setError('Please process an image first');
      return;
    }
  
    setIsDownloading(true);
    setError(null);
    const filename = files[0].name;
    const baseName = filename.replace('.tif', '');
  
    try {
      const zip = new JSZip();
      
      // 1. Add Sentinel GeoJSON
      const geojsonResponse = await fetch(
        `http://localhost:5000/download_sentinel_geojson/${baseName}.geojson`
      );
      if (!geojsonResponse.ok) throw new Error('Failed to download GeoJSON');
      zip.file(`${baseName}.geojson`, await geojsonResponse.blob());
  
      // 2. Add Processed Sentinel TIFF
      const processedResponse = await fetch(
        `http://localhost:5000/download_sentinel_processed_tif/${filename}`
      );
      if (!processedResponse.ok) throw new Error('Failed to download processed TIFF');
      zip.file(`${baseName}_processed.tif`, await processedResponse.blob());
  
      // 3. Add Original Sentinel TIFF (optional)
      try {
        const originalResponse = await fetch(
          `http://localhost:5000/download_sentinel_original_tif/${filename}`
        );
        if (originalResponse.ok) {
          zip.file(`${baseName}_original.tif`, await originalResponse.blob());
        }
      } catch (e) {
        console.warn('Original TIFF not included:', e);
      }
  
      // 4. Add Preview Image
      const base64Data = detectedImage.split(',')[1];
      zip.file(`${baseName}_preview.jpg`, base64Data, { base64: true });
  
      // Generate and download zip
      const content = await zip.generateAsync({ type: 'blob' });
      saveAs(content, `${baseName}_marine_debris_results.zip`);
  
    } catch (error) {
      setError(error.message || 'Failed to prepare download package');
      console.error('Download error:', error);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <>
      <NavBar />
      <div className="container mx-auto p-4">
        <div className="max-w-md mx-auto mb-8">
          <h2 className="text-2xl font-bold mb-4">Sentinel Data Upload</h2>

          <FileInput
            id="file-upload"
            className="mb-4"
            helperText="Upload a Sentinel .tif file"
            onChange={handleFileChange}
            accept=".tif,.tiff"
          />

          {error && <Alert color="failure" className="mb-4">{error}</Alert>}

          {isUploading && (
            <div className="mb-4">
              <p>Uploading... {uploadProgress.toFixed(0)}%</p>
            </div>
          )}

          <Button 
            onClick={handleUpload} 
            disabled={isUploading || files.length === 0}
            className="mb-8"
          >
            {isUploading ? 'Uploading...' : 'Upload File'}
          </Button>
        </div>

        {/* Results Display */}
        {detectedImage && (
          <div>
            <h2 className="text-xl font-bold mb-4">Detected Marine Debris</h2>
            <p className="mb-4 text-gray-600">({files[0].name}):</p>

            <div className="flex flex-col md:flex-row gap-8 mb-8">
              {/* Image Display */}
              <div className="flex-1">
                <img 
                  src={detectedImage} 
                  alt="Detection results" 
                  className="w-full h-auto border border-gray-300 rounded"
                />
              </div>

              {/* Detection Results Table */}
              <div className="flex-1">
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white border border-gray-300">
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
                      {detections[0]?.map((box, i) => {
                        const [id, area, x0, y0, x1, y1] = box.length >= 6 ? box : [null, null, null, null, null, null];
                        return (
                          <tr key={i} className="text-center hover:bg-gray-50">
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
                </div>
              </div>
            </div>

            <div className="text-center">
              <Button
                onClick={handleDownload}
                disabled={isDownloading}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
              >
                {isDownloading ? 'Processing...' : 'Download All Results (.zip)'}
              </Button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}