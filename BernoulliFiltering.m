function [trajectoryBernoulli,isValid] = BernoulliFiltering(trajectoryBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,currentTimeStep,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize)

isValid = true;

for timeStep = 1:length(measurementAssociation)
    numTrajectoryMixture = length(trajectoryBernoulli.trajectoryMixture);

    %prediction
    if trajectoryBernoulli.isAlive
        for l = 1:numTrajectoryMixture
            trajectoryBernoulli.trajectoryMixture(l).marginalMean = multiObjectDynamicModel.transitionMatrix*trajectoryBernoulli.trajectoryMixture(l).marginalMean;
            trajectoryBernoulli.trajectoryMixture(l).marginalCovariance = multiObjectDynamicModel.transitionMatrix*trajectoryBernoulli.trajectoryMixture(l).marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance;

            trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1)*multiObjectDynamicModel.survivalProbability;
            trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep-1)*(1-multiObjectDynamicModel.survivalProbability);
        end
        trajectoryBernoulli.endTime = trajectoryBernoulli.endTime + 1;
    end

    if measurementAssociation(timeStep) == 0
        %misdetection
        aliveProbability = 0;
        if trajectoryBernoulli.isAlive
            for l = 1:numTrajectoryMixture
                conditionalAliveProbability = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep);
                if conditionalAliveProbability*trajectoryBernoulli.existenceProbability >= minimumEndTimeProbability
                    trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep) = trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*(1-multiObjectMeasurementModel.detectionProbability);
                    trajectoryBernoulli.endTimeProbability(l,:) = trajectoryBernoulli.endTimeProbability(l,:)/(1-multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability);

                    aliveProbability = aliveProbability + multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability*trajectoryBernoulli.birthTimeProbability(l);
                end
            end
        end
        misDetectionLikelihood = log(1-trajectoryBernoulli.existenceProbability*aliveProbability);
        trajectoryBernoulli.likelihood = trajectoryBernoulli.likelihood + misDetectionLikelihood;
        trajectoryBernoulli.existenceProbability = trajectoryBernoulli.existenceProbability*(1-aliveProbability)/(1-trajectoryBernoulli.existenceProbability*aliveProbability);

        trajectoryBernoulli.associationHistory(currentTimeStep+timeStep) = 0;
    else
        %measurement update
        
        isInGate = false(numTrajectoryMixture,1);
        associationLikelihood = zeros(numTrajectoryMixture,1);
        if trajectoryBernoulli.isAlive
            measurement = measurements{currentTimeStep+timeStep}(:,measurementAssociation(timeStep));
            for l = 1:numTrajectoryMixture
                trajectory = trajectoryBernoulli.trajectoryMixture(l);
                if trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*trajectoryBernoulli.existenceProbability >= minimumEndTimeProbability
                    innovation = measurement-multiObjectMeasurementModel.observationMatrix*trajectory.marginalMean;
                    innovationCovariance = multiObjectMeasurementModel.observationMatrix*trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
                    innovationCovarianceInverse = eye(multiObjectMeasurementModel.measurementDimension)/innovationCovariance;
                    innovationCovarianceDeterminant = det(innovationCovariance);
                    mahalanobisDistance = innovation'*innovationCovarianceInverse*innovation;
                    if mahalanobisDistance < gateSize
                        isInGate(l) = true;

                        KalmanGain = trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix'*innovationCovarianceInverse;
                        trajectory.marginalMean = trajectory.marginalMean + KalmanGain*innovation;
                        marginalCovariance = (eye(multiObjectDynamicModel.stateDimension) - KalmanGain*multiObjectMeasurementModel.observationMatrix)*trajectory.marginalCovariance;
                        trajectory.marginalCovariance = (marginalCovariance+marginalCovariance')/2;

                        measurementLikelihood = exp(-mahalanobisDistance/2)/sqrt((2*pi)^multiObjectMeasurementModel.measurementDimension*innovationCovarianceDeterminant);
                        associationLikelihood(l) = trajectoryBernoulli.birthTimeProbability(l)*trajectoryBernoulli.endTimeProbability(l,currentTimeStep+timeStep)*multiObjectMeasurementModel.detectionProbability*measurementLikelihood;
                        probabilityVector = zeros(1,multiObjectDynamicModel.totalTimeSteps);
                        probabilityVector(currentTimeStep+timeStep) = 1;
                        trajectoryBernoulli.endTimeProbability(l,:) = probabilityVector;
                        trajectoryBernoulli.trajectoryMixture(l) = trajectory;
                    end
                end
            end
        end
        if any(isInGate)
            %remove mixture components with small birth time probability
            normalizedBirthProbability = associationLikelihood/sum(associationLikelihood);
            keepIndex = normalizedBirthProbability > minimumBirthTimeProbability;

            trajectoryBernoulli.birthTimeProbability = normalizedBirthProbability(keepIndex)/sum(normalizedBirthProbability(keepIndex));
            trajectoryBernoulli.birthTime = trajectoryBernoulli.birthTime(keepIndex);
            trajectoryBernoulli.endTime = trajectoryBernoulli.endTime(keepIndex);
            trajectoryBernoulli.endTimeProbability = trajectoryBernoulli.endTimeProbability(keepIndex,:);
            trajectoryBernoulli.trajectoryMixture = trajectoryBernoulli.trajectoryMixture(keepIndex);

            trajectoryBernoulli.likelihood = trajectoryBernoulli.likelihood + log(trajectoryBernoulli.existenceProbability*sum(associationLikelihood));

            trajectoryBernoulli.existenceProbability = 1;
            trajectoryBernoulli.associationHistory(currentTimeStep+timeStep) = measurementAssociation(timeStep);
        else
            isValid = false;
            break;
        end
    end

    [~,index] = max(trajectoryBernoulli.birthTimeProbability);
    if trajectoryBernoulli.endTimeProbability(index,currentTimeStep+timeStep)*trajectoryBernoulli.existenceProbability < minimumEndTimeProbability
        trajectoryBernoulli.isAlive = false;
    end

    pastInformation = trajectoryBernoulli;
    pastInformation.past = [];
    
    trajectoryBernoulli.past{timeStep+currentTimeStep} = pastInformation;
end

end