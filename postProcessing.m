newlyDetectedTrajectoryOriginal = newlyDetectedTrajectory;

%post-processing
for timeStep = 1:multiObjectDynamicModel.totalTimeSteps
    for i = 1:length(measurementAssociatedToNew{timeStep})
        if measurementAssociatedToNew{timeStep}(i)
            pastInformation = newlyDetectedTrajectory{timeStep}{i};
            newlyDetectedTrajectory{timeStep}{i}.past = cell(multiObjectDynamicModel.totalTimeSteps,1);
            newlyDetectedTrajectory{timeStep}{i}.past{timeStep} = pastInformation;
            newlyDetectedTrajectory{timeStep}{i} = BernoulliFiltering(newlyDetectedTrajectory{timeStep}{i},multiObjectDynamicModel,multiObjectMeasurementModel,measurements,zeros(1,multiObjectDynamicModel.totalTimeSteps-timeStep),timeStep,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
        end
    end
end

%extract trajectory multi-Bernoulli density from the most likely global hypothesis
trajectoryMultiBernoulli = [];
for i = 1:length(globalHypothesisMAP)
    if globalHypothesisMAP(i) > 0 && trajectoryBernoulli{i}(globalHypothesisMAP(i)).existenceProbability == 1
        localHypothesis = trajectoryBernoulli{i}(globalHypothesisMAP(i));
        currentTimeStep = localHypothesis.firstDetectedTimeStep;
        currentMeasurementAssociation = localHypothesis.associationHistory(currentTimeStep);
        measurementAssociation = localHypothesis.associationHistory(currentTimeStep+1:end);
        initialBernoulli = newlyDetectedTrajectory{currentTimeStep}{currentMeasurementAssociation}.past{currentTimeStep};
        initialBernoulli.past = newlyDetectedTrajectory{currentTimeStep}{currentMeasurementAssociation}.past;
        localHypothesisProcessed = BernoulliFiltering(initialBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,currentTimeStep,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
        trajectoryMultiBernoulli = [trajectoryMultiBernoulli localHypothesisProcessed];
    end
end

numBernoulli = length(trajectoryMultiBernoulli);
associationHistory = zeros(numBernoulli,multiObjectDynamicModel.totalTimeSteps);
for i = 1:numBernoulli
    associationHistory(i,:) = trajectoryMultiBernoulli(i).associationHistory;
end

for timeStep = 1:multiObjectDynamicModel.totalTimeSteps
    for i = 1:length(measurementAssociatedToNew{timeStep})
        if ~any(associationHistory(:,timeStep)==i) && measurementAssociatedToNew{timeStep}(i)
            trajectoryMultiBernoulli = [trajectoryMultiBernoulli newlyDetectedTrajectory{timeStep}{i}];
        end
    end
end