multiObjectDynamicModel.totalTimeSteps = 81;

%multi-object dynamic model
multiObjectDynamicModel.survivalProbability = 0.98;
multiObjectDynamicModel.stateDimension = 4;

%constant velocity motion model, [position-x, position-y, velocity-x, velocity-y]
samplingTime = 1;
multiObjectDynamicModel.transitionMatrix = kron([1 samplingTime;0 1],eye(2));
motionNoiseStd = 0.3;
multiObjectDynamicModel.motionNoiseCovariance = motionNoiseStd^2*kron([samplingTime^3/3 samplingTime^2/2;samplingTime^2/2 samplingTime],eye(2));

multiObjectDynamicModel.inverseMotionNoiseCovariance = multiObjectDynamicModel.motionNoiseCovariance\eye(multiObjectDynamicModel.stateDimension);
multiObjectDynamicModel.predictionInformationMatrix = [multiObjectDynamicModel.transitionMatrix.'; -eye(multiObjectDynamicModel.stateDimension)]*multiObjectDynamicModel.inverseMotionNoiseCovariance*[multiObjectDynamicModel.transitionMatrix -eye(multiObjectDynamicModel.stateDimension)];

maxTrajectoryLength = multiObjectDynamicModel.stateDimension*multiObjectDynamicModel.totalTimeSteps;
initialInformationVector = zeros(maxTrajectoryLength,1);
initialInformationMatrix = spalloc(maxTrajectoryLength,maxTrajectoryLength,maxTrajectoryLength*multiObjectDynamicModel.stateDimension*(multiObjectDynamicModel.stateDimension-1));

%multi-object measurement model
multiObjectMeasurementModel.detectionProbability = 0.7;

multiObjectMeasurementModel.measurementDimension = 2;
multiObjectMeasurementModel.observationMatrix = [eye(2) zeros(2)];
measurementNoiseStd = 1;
multiObjectMeasurementModel.measurementNoiseCovariance = measurementNoiseStd^2*eye(multiObjectMeasurementModel.measurementDimension);
multiObjectMeasurementModel.inverseMeasurementNoiseCovariance = multiObjectMeasurementModel.measurementNoiseCovariance\eye(multiObjectMeasurementModel.measurementDimension);

squareSideLength = 200;
multiObjectMeasurementModel.rangeOfInterest = [-squareSideLength squareSideLength;-squareSideLength squareSideLength];
multiObjectMeasurementModel.areaOfInterest = (2*squareSideLength)^2;

multiObjectMeasurementModel.PoissonClutterRate = 30;
multiObjectMeasurementModel.PoissonClutterIntensity = multiObjectMeasurementModel.PoissonClutterRate/multiObjectMeasurementModel.areaOfInterest;