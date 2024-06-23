function inGate = ellipsoidalGatingforBernoulliPrecompute(numTrajectoryMixture,predictedMean,inverseInnovationCovariance,measurements,measurementAssociation,gateSize)

if measurementAssociation(1) == 0
    inGate = true;
    return;
end

measurement = measurements(:,measurementAssociation(1));
inGate = false;
innovation = measurement-predictedMean;
for l = 1:numTrajectoryMixture
    mahalanobisDistance = innovation(:,l)'*inverseInnovationCovariance(:,:,l)*innovation(:,l);
    if mahalanobisDistance < gateSize
        inGate = true;
        break;
    end
end

end