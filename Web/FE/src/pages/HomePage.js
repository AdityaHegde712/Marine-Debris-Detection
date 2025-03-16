import { Models } from "../components/Models";
import { NavBar } from "../components/NavBar";
import "../styles/Video.css";
import { HR } from "flowbite-react";
import { useEffect, useRef } from "react";
import { FooterComponent } from "../components/FooterComponent";

export function HomePage() {
  const servicesRef = useRef(null);

  // Function to scroll to the services section
  const scrollToServices = () => {
    servicesRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            servicesRef.current?.classList.add("fade-in");
          } else {
            servicesRef.current?.classList.remove("fade-in");
          }
        });
      },
      { threshold: 0.5 }
    );

    if (servicesRef.current) {
      observer.observe(servicesRef.current);
    }

    return () => {
      if (servicesRef.current) {
        observer.unobserve(servicesRef.current);
      }
    };
  }, []);

  return (
    <>
      {/* Pass scroll function to NavBar */}
      <NavBar onGetStartedClick={scrollToServices} />
      
      <div className="video-container padding:40px">
        <video className="homepage-video" style={{ padding: "40px" }} autoPlay loop muted>
          <source src="/assets/1890-151167947_small.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div
          className="video-text"
          style={{
            position: "absolute",
            zIndex: "10",
            color: "white",
            backgroundColor: "rgba(0, 0, 0, 0.01)",
            padding: "30px",
            left: "480px",
          }}
        >
          Leveraging AI and computer vision to accurately detect marine debris, aiding environmental protection and cleanup efforts.
        </div>
      </div>

      <HR />

      <p className="text-gray-500 dark:text-gray-400" style={{ padding: "30px" }}>
        The prevalence and impact of marine debris are escalating worldwide, posing significant threats to marine ecosystems and coastal communities. Countless habitats and species are affected as pollution disrupts marine life and endangers biodiversity.
      </p>

      {/* Discover Our Services Section */}
      <div style={{ padding: "40px", backgroundColor: "#207082" }} ref={servicesRef}>
        <p style={{ color: "white", padding: "10px", fontSize: "30px" }}>Discover Our Services</p>
        <Models />
      </div>

      <FooterComponent />
    </>
  );
}
