"use client";
import { Button, Navbar } from "flowbite-react";
import { useNavigate, useLocation } from "react-router";

export function NavBar({ onGetStartedClick }) {
  const navigate = useNavigate();
  const location = useLocation();

  const handlehome = () => {
    navigate("/homepage");
  };

  return (
    <Navbar
      fluid
      rounded
      style={{
        border: "1px solid #d1d5db", // Light gray border
      }}
    >
      <h1>Marine Debris Detection</h1>

      <div className="flex md:order-2">
        {/* Only show the button if we're on the homepage */}
        {location.pathname === "/homepage" && (
          <Button
            gradientDuoTone="purpleToBlue"
            style={{ cursor: "pointer", fontSize: "1.23rem" }}
            onClick={onGetStartedClick}
          >
            Get started
          </Button>
        )}
        <Navbar.Toggle />
      </div>

      <Navbar.Collapse>
        <Navbar.Link onClick={handlehome} active style={{ cursor: "pointer", fontSize: "1.25rem" }}>
          Home
        </Navbar.Link>
      </Navbar.Collapse>
    </Navbar>
  );
}