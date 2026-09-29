# Missing Temperature Sensor Data Reconstruction

## Project Overview

This project investigates the reconstruction of missing temperature sensor measurements using machine learning regression methods.

The project was completed for MMA3001 Numerical Methods and Machine Learning.

Two regression approaches were investigated:

- Linear Regression
- Decision Tree Regression

The aim is to estimate a missing temperature measurement using surrounding temperature measurements recorded before and after the missing point.

---

## Engineering Problem

Building environmental monitoring systems rely on continuous sensor measurements.

However, sensor data may occasionally be missing because of communication issues, temporary sensor failure, or data recording problems.

Missing measurements can reduce the usefulness of environmental monitoring data.

This project investigates whether surrounding temperature measurements can be used to reconstruct a missing measurement accurately.

The reconstruction considered in this project is an offline reconstruction problem rather than future temperature forecasting.

---

## Dataset

The project uses environmental sensor data provided for MMA3001.

The original dataset contains several environmental variables and multiple sensors.

For this project, only temperature measurements were extracted.

The selected sensor was:

```text
Sensor ID: 6012002000869