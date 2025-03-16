// import React, { useState } from 'react';
// import { FileInput, Button, Spinner, Progress } from 'flowbite-react';
// import { NavBar } from '../components/NavBar';
// import { FooterComponent } from '../components/FooterComponent';
// // import {HR } from "flowbite-react";
// import '../styles/Footer.css';
// import axios from 'axios';
// export default function PredictFire(){
//   const [file, setFile] = useState(null);
//   const [uploadProgress, setUploadProgress] = useState(0);
//   const [isUploading, setIsUploading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleFileChange = (e) => {
//     setFile(e.target.files[0]);
//   };

//   // const handleUpload = async () => {
//   //   if (!file) {
//   //     setError('Please select a file to upload');
//   //     return;
//   //   }

//   //   setIsUploading(true);
//   //   setError(null);

//   //   const formData = new FormData();
//   //   formData.append('file', file);

//   //   try {
//   //     // Replace this after model deploymenyt with your backend API endpoint
//   //     const uploadUrl = 'http://127.0.0.1:5000/predict/wildfire';

//   //     const response = await fetch(uploadUrl, {
//   //       method: 'POST',
//   //       body: formData,
//   //       headers: {
//   //         // Add any headers here if needed, such as authorization tokens
//   //       },
//   //       onUploadProgress: (progressEvent) => {
//   //         const percentCompleted = Math.round(
//   //           (progressEvent.loaded * 100) / progressEvent.total
//   //         );
//   //         setUploadProgress(percentCompleted);
//   //       },
//   //     });

//   //     if (!response.ok) {
//   //       throw new Error('Upload failed');
//   //     }

//   //     setIsUploading(false);
//   //     alert('File uploaded successfully!');
//   //   } catch (error) {
//   //     setIsUploading(false);
//   //     setError(error.message);
//   //   }
//   // };



// const handleUpload = async () => {
//     if (!file) {
//         setError('Please select a file to upload');
//         return;
//     }

//     setIsUploading(true);
//     setError(null);

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//         const response = await axios.post('http://127.0.0.1:5000/predict/wildfire', formData, {
//             onUploadProgress: (progressEvent) => {
//                 const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
//                 setUploadProgress(percentCompleted);
//             },
//         });

//         setIsUploading(false);
//         alert('File uploaded successfully!');
//         console.log(response.data); // Display the prediction result
//     } catch (error) {
//         setIsUploading(false);
//         setError(error.message);
//     }
// };

//   return (
//     <>
//     <NavBar />
//     <div className="max-w-md mx-auto mt-10">
//       <h2 className="text-2xl font-bold mb-4">Wildfire Prediction - File Upload</h2>

//       <FileInput 
//         id="large-file-upload"
//         className="mb-4"
//         helperText="Please upload your dataset (e.g., images, CSV, etc.)"
//         onChange={handleFileChange}
//       />

//       {error && <p className="text-red-500">{error}</p>}

//       {isUploading && (
//         <div className="mb-4">
//           <Spinner aria-label="File uploading..." />
//           <p>Uploading... {uploadProgress}%</p>
//           <Progress progress={uploadProgress} />
//         </div>
//       )}

//       <Button onClick={handleUpload} disabled={isUploading || !file}>
//         {isUploading ? 'Uploading...' : 'Upload File'}
//       </Button>
//       <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
//       <Button gradientDuoTone="tealToLime" >Download Predictions</Button>
//       </div>
//     </div>
//     {/* <div className='footer'>
//     <FooterComponent />
//     </div> */}
//     </>
//   );
// };

// -----------------------------------------------------------------------------------------------

import React, { useState } from 'react';
import { FileInput, Button, Spinner, Progress } from 'flowbite-react';
import { NavBar } from '../components/NavBar';
import { FooterComponent } from '../components/FooterComponent';
import '../styles/Footer.css';
import axios from 'axios';

export default function PredictFire(){
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState(null);  // Store prediction response here

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
        const response = await axios.post('http://127.0.0.1:5000/predict/wildfire', formData, {
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setUploadProgress(percentCompleted);
            },
        });

        setIsUploading(false);
        setPredictionData(response.data);  // Store the prediction data in state
    } catch (error) {
        setIsUploading(false);
        setError(error.message);
    }
  };

  const handleDownload = () => {
    if (predictionData) {
      const jsonData = JSON.stringify(predictionData, null, 2);
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = 'prediction.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <>
      <NavBar />
      <div className="max-w-md mx-auto mt-10">
        <h2 className="text-2xl font-bold mb-4">Wildfire Prediction - File Upload</h2>

        <FileInput 
          id="large-file-upload"
          className="mb-4"
          helperText="Please upload your dataset (e.g., images, CSV, etc.)"
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

        {/* Display prediction data as a table if available */}
        {predictionData && (
          <div className="mt-6">
            <h3 className="text-xl font-bold">Prediction Results</h3>
            <table className="table-auto w-full mt-4">
              <thead>
                <tr>
                  <th className="px-4 py-2">Prediction</th>
                  {/* <th className="px-4 py-2">Probability</th> */}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2" style={{ textAlign: 'center'}}>{predictionData.prediction}</td>
                  {/* <td className="border px-4 py-2">{predictionData.probability.toFixed(4)}</td> */}
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* Download Button */}
        <div className="mt-4 flex justify-center">
          <Button gradientDuoTone="tealToLime" onClick={handleDownload} disabled={!predictionData}>
            Download Predictions
          </Button>
        </div>
      </div>
    </>
  );
}
