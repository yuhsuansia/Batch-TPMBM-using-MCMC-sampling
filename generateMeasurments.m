%generate measurements
measurements = cell(multiObjectDynamicModel.totalTimeSteps,1);
%generate object detections
for i = 1:length(objectTrajectory)
    for timeStep = objectTrajectory(i).birthTime:objectTrajectory(i).endTime
        if rand < multiObjectMeasurementModel.detectionProbability
            objectDetection = mvnrnd((multiObjectMeasurementModel.observationMatrix*objectTrajectory(i).stateSequence(:,timeStep-objectTrajectory(i).birthTime+1))',multiObjectMeasurementModel.measurementNoiseCovariance)';
            measurements{timeStep} = [measurements{timeStep} objectDetection];
        end
    end
end

%generate clutter
for timeStep = 1:multiObjectDynamicModel.totalTimeSteps
    numClutter = poissrnd(multiObjectMeasurementModel.PoissonClutterRate);
    clutterMeasurements = multiObjectMeasurementModel.rangeOfInterest(:,1) + (multiObjectMeasurementModel.rangeOfInterest(:,2)-multiObjectMeasurementModel.rangeOfInterest(:,1)).*rand(2,numClutter);
    measurements{timeStep} = [measurements{timeStep} clutterMeasurements];
end