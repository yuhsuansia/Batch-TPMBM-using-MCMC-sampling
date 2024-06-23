function inGate = ellipsoidalGatingforBernoulli(trajectoryBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,currentTimeStep,gateSize)

if measurementAssociation(1) == 0
    inGate = true;
    return;
end

measurement = measurements{currentTimeStep+1}(:,measurementAssociation(1));
numTrajectoryMixture = length(trajectoryBernoulli.trajectoryMixture);
inGate = false;
for l = 1:numTrajectoryMixture
    innovation = measurement-multiObjectMeasurementModel.observationMatrix*(multiObjectDynamicModel.transitionMatrix*trajectoryBernoulli.trajectoryMixture(l).marginalMean);
    innovationCovariance = multiObjectMeasurementModel.observationMatrix*(multiObjectDynamicModel.transitionMatrix*trajectoryBernoulli.trajectoryMixture(l).marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance)*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
    mahalanobisDistance = innovation'/innovationCovariance*innovation;
    if mahalanobisDistance < gateSize
        inGate = true;
        break;
    end
end

end