function [meanSmoothSequence,covarianceSmoothSequence] = RTSsmoothing(meanFilterSequence,covarianceFilterSequence,multiObjectDynamicModel)

meanSmoothSequence = meanFilterSequence;
covarianceSmoothSequence = covarianceFilterSequence;

sequenceLength = size(meanFilterSequence,2);
if sequenceLength > 1
    for k = sequenceLength-1:-1:1
        meanPrediction = multiObjectDynamicModel.transitionMatrix*meanFilterSequence(:,k);
        covariancePrediction = multiObjectDynamicModel.transitionMatrix*covarianceFilterSequence(:,:,k)*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance;
        smoothGain = covarianceFilterSequence(:,:,k)*multiObjectDynamicModel.transitionMatrix'/covariancePrediction;
        meanSmoothSequence(:,k) = meanFilterSequence(:,k) + smoothGain*(meanSmoothSequence(:,k+1) - meanPrediction);
        covarianceSmoothSequence(:,:,k) = covarianceFilterSequence(:,:,k) + smoothGain*(covarianceSmoothSequence(:,:,k+1) - covariancePrediction)*smoothGain';
    end
end

end