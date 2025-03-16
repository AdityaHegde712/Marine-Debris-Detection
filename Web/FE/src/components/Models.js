import { Card, Button } from "flowbite-react";
import { useNavigate } from "react-router-dom";  // Import the useNavigate hook

export function Models() {
  const navigate = useNavigate();  // Initialize navigate hook

  const handleObjectDetectionReadMore = () => {
    navigate("/object-detection");  // Navigate to the Object Detection component
  };
  const handleSentinelReadMore = () => {
    navigate("/sentinel");  // Navigate to the Sentinel component
  };
  // const handlePredictFireReadMore = () => {
  //   navigate("/predict-fire");  // Navigate to the BurnScar component
  // };

  return (
    <div className="flex flex-wrap justify-center gap-3">
      <Card className="max-w-sm">
      <h5 class="text-2xl font-bold tracking-tight text-gray-900 dark:text-white text-center">
  Using Object Detection
</h5>

        <p className="font-normal text-gray-700 dark:text-gray-400 text-center">
          Using Object Detection
        </p>
        <Button gradientDuoTone="tealToLime" pill onClick={handleObjectDetectionReadMore}>  {/* Navigate to BurnScar on click */}
          Read more
        </Button>
      </Card>

      {/* Land Cover Classification Card */}
      <Card className="max-w-sm">
        <h5 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white text-center">
          Detection on Sentinel Data
        </h5>
        <p className="font-normal text-gray-700 dark:text-gray-400 text-center">
          Using Satellite Imagery Analysis
        </p>
        <Button  gradientDuoTone="tealToLime" pill onClick={handleSentinelReadMore}>
          Read more
        </Button>
      </Card>

      {/* Wildfire Prediction Card */}
      {/* <Card className="max-w-sm">
        <h5 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
          Wildfire Prediction
        </h5>
        <p className="font-normal text-gray-700 dark:text-gray-400">
          Using Predictive Models
        </p>
        <Button color="failure" pill onClick={handlePredictFireReadMore}>
          Read more
        </Button>
      </Card> */}
    </div>
  );
}
