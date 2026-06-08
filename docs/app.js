(function () {
  const regionProfiles = {
    AIIMS: { pm25: -1.5, pm10: -4, so2: 0.2, label: "AIIMS" },
    BHATAGAON: { pm25: 4.6, pm10: 8, so2: 0.7, label: "Bhatagaon" },
    IGKV: { pm25: -3.8, pm10: -9, so2: -0.4, label: "IGKV" },
    SILTARA: { pm25: 3.2, pm10: 12, so2: 0.9, label: "SILTARA" },
  };

  const presets = {
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
      pm10: 214,
      so2: 34,
      temp: 35,
      humidity: 68,
      wind: 1.8,
      hour: 22,
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
      return { label: "Severe", tone: "#f1a8a8" };
    }
    if (prediction.pm25 > 60 || prediction.pm10 > 160 || prediction.so2 > 28) {
      return { label: "Poor", tone: "#f0c66c" };
    }
    if (prediction.pm25 > 35 || prediction.pm10 > 100 || prediction.so2 > 18) {
      return { label: "Moderate", tone: "#bfe7d9" };
    }
    return { label: "Good", tone: "#8ce0d7" };
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

  function setBar(id, value, max) {
    const element = document.getElementById(id);
    element.style.width = `${clamp((value / max) * 100, 4, 100)}%`;
  }

  function renderPrediction() {
    const prediction = calculatePrediction(readInput());
    const risk = classifyRisk(prediction);
    const riskLabel = document.getElementById("risk-label");

    document.getElementById("pred-pm25").textContent = round1(prediction.pm25);
    document.getElementById("pred-pm10").textContent = round1(prediction.pm10);
    document.getElementById("pred-so2").textContent = round1(prediction.so2);

    riskLabel.textContent = risk.label;
    riskLabel.style.background = risk.tone;

    setBar("bar-pm25", prediction.pm25, 120);
    setBar("bar-pm10", prediction.pm10, 300);
    setBar("bar-so2", prediction.so2, 60);

    document.getElementById("prediction-summary").textContent =
      `${prediction.regionLabel} next-step forecast: PM2.5 ${round1(prediction.pm25)}, ` +
      `PM10 ${round1(prediction.pm10)}, SO2 ${round1(prediction.so2)}. ` +
      `The overall air-quality condition is estimated as ${risk.label.toLowerCase()}.`;
  }

  function applyPreset(name) {
    const preset = presets[name];
    if (!preset) return;

    Object.entries(preset).forEach(([key, value]) => {
      const element = document.getElementById(key);
      if (element) element.value = value;
    });
    renderPrediction();
  }

  fields.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener("input", renderPrediction);
      element.addEventListener("change", renderPrediction);
    }
  });

  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });

  renderPrediction();
})();
