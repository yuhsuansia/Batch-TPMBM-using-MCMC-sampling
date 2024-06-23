function trajectoryBernoulli = BernoulliFilteringSimplified(trajectoryBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,currentTimeStep,minimumEndTimeProbability)

for timeStep = 1:length(measurementAssociation)
    numTrajectoryMixture = length(trajectoryBernoulli.trajectoryMixture);

    %prediction
    for l = 1:numTrajectoryMixture
        trajectory = trajectoryBernoulli.trajectoryMixture(l);
        if trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1)*trajectoryBernoulli.existenceProbability >= minimumEndTimeProbability
            trajectory.marginalMean = multiObjectDynamicModel.transitionMatrix*trajectory.marginalMean;
            trajectory.marginalCovariance = multiObjectDynamicModel.transitionMatrix*trajectory.marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance;

            trajectoryBernoulli.endTime(l) = trajectoryBernoulli.endTime(l) + 1;
            trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1)*multiObjectDynamicModel.survivalProbability;
            trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1)*(1-multiObjectDynamicModel.survivalProbability);
        end
        trajectoryBernoulli.trajectoryMixture(l) = trajectory;
    end

    if measurementAssociation(timeStep) == 0
        %misdetection
        aliveProbability = 0;
        for l = 1:numTrajectoryMixture
            trajectory = trajectoryBernoulli.trajectoryMixture(l);
            conditionalAliveProbability = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep);
            if conditionalAliveProbability*trajectoryBernoulli.existenceProbability >= minimumEndTimeProbability
                trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*(1-multiObjectMeasurementModel.detectionProbability);
                trajectoryBernoulli.endTimeProbability(l,:) = trajectoryBernoulli.endTimeProbability(l,:)/(1-multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability);

                trajectory.filterMean(:,currentTimeStep+timeStep) = trajectory.marginalMean;
                trajectory.filterCovariance(:,:,currentTimeStep+timeStep) = trajectory.marginalCovariance;

                aliveProbability = aliveProbability + multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability*trajectoryBernoulli.birthTimeProbability(l);
                trajectoryBernoulli.trajectoryMixture(l) = trajectory;
            end
        end
        trajectoryBernoulli.existenceProbability = trajectoryBernoulli.existenceProbability*(1-aliveProbability)/(1-trajectoryBernoulli.existenceProbability*aliveProbability);

    else
        %measurement update

        associationLikelihood = zeros(numTrajectoryMixture,1);
        measurement = measurements{currentTimeStep+timeStep}(:,measurementAssociation(timeStep));
        for l = 1:numTrajectoryMixture
            trajectory = trajectoryBernoulli.trajectoryMixture(l);
            if trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*trajectoryBernoulli.existenceProbability >= minimumEndTimeProbability
                innovation = measurement-multiObjectMeasurementModel.observationMatrix*trajectory.marginalMean;
                innovationCovariance = multiObjectMeasurementModel.observationMatrix*trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
                innovationCovarianceInverse = eye(multiObjectMeasurementModel.measurementDimension)/innovationCovariance;
                innovationCovarianceDeterminant = det(innovationCovariance);
                mahalanobisDistance = innovation'*innovationCovarianceInverse*innovation;

                KalmanGain = trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix'*innovationCovarianceInverse;
                trajectory.marginalMean = trajectory.marginalMean + KalmanGain*innovation;
                marginalCovariance = (eye(multiObjectDynamicModel.stateDimension) - KalmanGain*multiObjectMeasurementModel.observationMatrix)*trajectory.marginalCovariance;
                trajectory.marginalCovariance = (marginalCovariance+marginalCovariance')/2;

                trajectory.filterMean(:,currentTimeStep+timeStep) = trajectory.marginalMean;
                trajectory.filterCovariance(:,:,currentTimeStep+timeStep) = trajectory.marginalCovariance;
    
                measurementLikelihood = exp(-mahalanobisDistance/2)/sqrt((2*pi)^multiObjectMeasurementModel.measurementDimension*innovationCovarianceDeterminant);
                associationLikelihood(l) = trajectoryBernoulli.birthTimeProbability(l)*trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*multiObjectMeasurementModel.detectionProbability*measurementLikelihood;
                trajectoryBernoulli.endTimeProbability(l,:) = zeros(1,multiObjectDynamicModel.totalTimeSteps);
                trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep) = 1;
                trajectoryBernoulli.trajectoryMixture(l) = trajectory;
            end
        end
        trajectoryBernoulli.birthTimeProbability = associationLikelihood/sum(associationLikelihood);
        trajectoryBernoulli.existenceProbability = 1;
    end
end

end