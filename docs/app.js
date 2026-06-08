(function () {
  const regionProfiles = {
    AIIMS: { pm25: -1.5, pm10: -4, so2: 0.2, label: "AIIMS" },
    BHATAGAON: { pm25: 4.6, pm10: 8, so2: 0.7, label: "Bhatagaon" },
    IGKV: { pm25: -3.8, pm10: -9, so2: -0.4, label: "IGKV" },
    SILTARA: { pm25: 3.2, pm10: 12, so2: 0.9, label: "SILTARA" },
  };

  const presets = {
    siltara: {
      region: "SILTARA",
      pm25: 78,
      pm10: 145,
      so2: 14,
      temp: 31,
      humidity: 62,
      wind: 2.1,
      hour: 8,
      month: 12,
    },
    clean: {
      region: "IGKV",
      pm25: 18,
      pm10: 54,
      so2: 8,
      temp: 29,
      humidity: 42,
      wind: 5.6,
      hour: 11,
      month: 2,
    },
    traffic: {
      region: "AIIMS",
      pm25: 46,
      pm10: 138,
      so2: 21,
      temp: 32,
      humidity: 61,
      wind: 2.6,
      hour: 19,
      month: 11,
    },
    industrial: {
      region: "SILTARA",
      pm25: 78,
      pm10: 145,
      so2: 14,
      temp: 31,
      humidity: 62,
      wind: 2.1,
      hour: 8,
      month: 12,
    },
  };

  const fields = ["region", "pm25", "pm10", "so2", "temp", "humidity", "wind", "hour", "month"];

  function getNumber(id) {
    const element = document.getElementById(id);
    return Number.parseFloat(element.value) || 0;
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function round1(value) {
    return (Math.round(value * 10) / 10).toFixed(1);
  }

  function setText(id, text) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = text;
    }
  }

  function calculatePrediction(input) {
    const profile = regionProfiles[input.region] || regionProfiles.AIIMS;
    const rushHour = input.hour >= 17 && input.hour <= 21 ? 1 : input.hour >= 7 && input.hour <= 10 ? 0.7 : 0;
    const nightStagnation = input.hour >= 22 || input.hour <= 5 ? 0.55 : 0;
    const winter = input.month === 11 || input.month === 12 || input.month <= 2 ? 1 : 0;
    const humidityEffect = Math.max(0, input.humidity - 55) / 18;
    const windRelief = Math.min(input.wind, 10) / 4;
    const heatLift = Math.max(0, input.temp - 32) / 10;

    const pm25 =
      input.pm25 * 0.76 +
      input.pm10 * 0.055 +
      profile.pm25 +
      rushHour * 4.5 +
      nightStagnation * 3.2 +
      winter * 5.8 +
      humidityEffect * 2.4 -
      windRelief * 2.8;

    const pm10 =
      input.pm10 * 0.79 +
      input.pm25 * 0.22 +
      profile.pm10 +
      rushHour * 9.5 +
      winter * 11.0 +
      humidityEffect * 4.2 -
      windRelief * 5.6 +
      heatLift * 3.5;

    const so2 =
      input.so2 * 0.82 +
      profile.so2 +
      rushHour * 0.9 +
      nightStagnation * 0.6 +
      winter * 0.8 +
      heatLift * 0.5 -
      windRelief * 0.45;

    return {
      pm25: clamp(pm25, 0, 500),
      pm10: clamp(pm10, 0, 700),
      so2: clamp(so2, 0, 120),
      regionLabel: profile.label,
    };
  }

  function classifyRisk(prediction) {
    if (prediction.pm25 > 90 || prediction.pm10 > 250 || prediction.so2 > 40) {
      return { label: "Severe", varPrefix: "severe" };
    }
    if (prediction.pm25 > 60 || prediction.pm10 > 160 || prediction.so2 > 28) {
      return { label: "Poor", varPrefix: "poor" };
    }
    if (prediction.pm25 > 35 || prediction.pm10 > 100 || prediction.so2 > 18) {
      return { label: "Moderate", varPrefix: "moderate" };
    }
    return { label: "Good", varPrefix: "good" };
  }

  function readInput() {
    return {
      region: document.getElementById("region").value,
      pm25: getNumber("pm25"),
      pm10: getNumber("pm10"),
      so2: getNumber("so2"),
      temp: getNumber("temp"),
      humidity: getNumber("humidity"),
      wind: getNumber("wind"),
      hour: getNumber("hour"),
      month: getNumber("month"),
    };
  }

  function setGauge(id, value, max, prefix) {
    const element = document.getElementById(`gauge-${id}`);
    if (element) {
      const pct = clamp(value / max, 0, 1);
      const circumference = 251.2;
      const offset = circumference * (1 - pct);
      element.style.strokeDashoffset = offset;
      element.style.stroke = `var(--color-${prefix}-start)`;
    }
  }

  function setBar(id, value, max, prefix) {
    const element = document.getElementById(id);
    if (element) {
      element.style.width = `${clamp((value / max) * 100, 4, 100)}%`;
      element.style.background = `linear-gradient(90deg, var(--color-${prefix}-start), var(--color-${prefix}-end))`;
    }
  }

  function buildAnomalySummary(input, prediction) {
    const alerts = [];
    if (input.pm25 >= 75 || prediction.pm25 >= 75) {
      alerts.push(`PM2.5 elevated at ${round1(Math.max(input.pm25, prediction.pm25))} ug/m3`);
    }
    if (input.pm10 >= 145 || prediction.pm10 >= 145) {
      alerts.push(`PM10 spike watch at ${round1(Math.max(input.pm10, prediction.pm10))} ug/m3`);
    }
    if (input.so2 >= 28 || prediction.so2 >= 28) {
      alerts.push(`SO2 elevated at ${round1(Math.max(input.so2, prediction.so2))} ug/m3`);
    }
    return alerts;
  }

  function renderSupportPanels(input, prediction, risk) {
    const alerts = buildAnomalySummary(input, prediction);
    const primaryDriver = prediction.pm10 >= prediction.pm25 * 1.8 ? "PM10 lag features" : "PM2.5 trend features";
    const weatherSignal = input.wind <= 2.5 ? "Low wind stagnation" : "Wind dispersion relief";
    const anomalyStatus = alerts.length ? "Spike watch active" : "No spike detected";
    const anomalyCopy = alerts.length ? alerts.join("; ") : "Current readings stay below the active alert thresholds.";
    const reportLines = [
      "AirSense AI report",
      `Region: ${prediction.regionLabel}`,
      `Inputs: PM2.5 ${round1(input.pm25)} ug/m3, PM10 ${round1(input.pm10)} ug/m3, SO2 ${round1(input.so2)} ug/m3`,
      `Forecast: PM2.5 ${round1(prediction.pm25)} ug/m3, PM10 ${round1(prediction.pm10)} ug/m3, SO2 ${round1(prediction.so2)} ug/m3`,
      `AQI-style risk: ${risk.label}`,
      `Anomaly review: ${anomalyCopy}`,
      "Recommendation: Use the Streamlit app for the artifact-backed forecast, downloadable report, and model explanation.",
    ];

    setText("top-factor", `${primaryDriver} + ${prediction.regionLabel}`);
    setText("factor-copy", `${primaryDriver} and station context are the strongest forecast drivers.`);
    setText("inline-anomaly-status", anomalyStatus);
    setText("inline-anomaly-copy", anomalyCopy);
    setText("explain-primary", primaryDriver);
    setText("explain-primary-copy", `${primaryDriver} carry recent pollutant behavior into the next-step forecast.`);
    setText("explain-context", `Region: ${prediction.regionLabel}`);
    setText("explain-weather", weatherSignal);
    setText("explain-weather-copy", input.wind <= 2.5 ? "Low wind speed reduces dispersion and raises review priority." : "Higher wind speed lowers the stagnation signal in this scenario.");
    setText("anomaly-status", anomalyStatus);
    setText("anomaly-copy", anomalyCopy);
    setText("report-risk", `${risk.label} risk`);
    setText("report-text", reportLines.join("\n"));
  }

  function renderPrediction() {
    const input = readInput();
    const prediction = calculatePrediction(input);
    const risk = classifyRisk(prediction);
    const riskLabel = document.getElementById("risk-label");
    const prefix = risk.varPrefix;

    document.getElementById("pred-pm25").textContent = round1(prediction.pm25);
    document.getElementById("pred-pm10").textContent = round1(prediction.pm10);
    document.getElementById("pred-so2").textContent = round1(prediction.so2);

    riskLabel.textContent = risk.label;
    riskLabel.style.background = `linear-gradient(135deg, var(--color-${prefix}-start), var(--color-${prefix}-end))`;
    riskLabel.style.color = prefix === "good" || prefix === "moderate" ? "#060913" : "#ffffff";
    riskLabel.style.boxShadow = `0 4px 12px var(--color-${prefix}-glow)`;

    setGauge("pm25", prediction.pm25, 120, prefix);
    setGauge("pm10", prediction.pm10, 300, prefix);
    setGauge("so2", prediction.so2, 60, prefix);

    setBar("bar-pm25", prediction.pm25, 120, prefix);
    setBar("bar-pm10", prediction.pm10, 300, prefix);
    setBar("bar-so2", prediction.so2, 60, prefix);

    const tips = {
      good: "Air quality is satisfactory. Great time for outdoor activities!",
      moderate: "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
      poor: "Wear a mask outdoors. Sensitive groups may experience health effects.",
      severe: "Avoid outdoor activities. Keep windows closed and run air purifiers."
    };

    document.getElementById("prediction-summary").innerHTML =
      `<strong>${prediction.regionLabel} next-step forecast:</strong> PM2.5 is predicted at <strong>${round1(prediction.pm25)}</strong> ug/m3, ` +
      `PM10 at <strong>${round1(prediction.pm10)}</strong> ug/m3, and SO2 at <strong>${round1(prediction.so2)}</strong> ug/m3.<br>` +
      `<span style="margin-top: 8px; display: inline-block; font-size: 0.9em; opacity: 0.95;"><strong>Health recommendation:</strong> ${tips[prefix]}</span>`;

    renderSupportPanels(input, prediction, risk);
  }

  function updateSliderDisplay(id, value) {
    const display = document.getElementById(`val-${id}`);
    if (display) {
      if (id === "wind" || id === "temp" || id === "so2") {
        display.textContent = Number(value).toFixed(1);
      } else {
        display.textContent = Math.round(Number(value));
      }
    }
  }

  function applyPreset(name) {
    const preset = presets[name];
    if (!preset) return;

    Object.entries(preset).forEach(([key, value]) => {
      const element = document.getElementById(key);
      if (element) {
        element.value = value;
        updateSliderDisplay(key, value);
      }
    });
    renderPrediction();
  }

  // Set up theme toggler
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    const sunIcon = themeToggle.querySelector(".sun-icon");
    const moonIcon = themeToggle.querySelector(".moon-icon");

    function setTheme(theme) {
      if (theme === "light") {
        document.body.classList.add("light-theme");
        if (sunIcon) sunIcon.style.display = "none";
        if (moonIcon) moonIcon.style.display = "block";
      } else {
        document.body.classList.remove("light-theme");
        if (sunIcon) sunIcon.style.display = "block";
        if (moonIcon) moonIcon.style.display = "none";
      }
      localStorage.setItem("theme", theme);
    }

    themeToggle.addEventListener("click", () => {
      const isLight = document.body.classList.contains("light-theme");
      setTheme(isLight ? "dark" : "light");
    });

    const savedTheme = localStorage.getItem("theme") || "dark";
    setTheme(savedTheme);
  }

  // Set up slider display and event listeners
  fields.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      if (id !== "region") {
        updateSliderDisplay(id, element.value);
      }

      const handler = () => {
        if (id !== "region") {
          updateSliderDisplay(id, element.value);
        }
        renderPrediction();
      };

      element.addEventListener("input", handler);
      element.addEventListener("change", handler);
    }
  });

  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });

  // Track active section in header nav on scroll
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll("header nav a");

  window.addEventListener("scroll", () => {
    let current = "";
    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;
      if (pageYOffset >= sectionTop - 120) {
        current = section.getAttribute("id");
      }
    });

    navLinks.forEach((a) => {
      a.classList.remove("active");
      if (a.getAttribute("href").slice(1) === current) {
        a.classList.add("active");
      }
    });
  });

  renderPrediction();
})();
