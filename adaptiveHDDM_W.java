package IJCNNMDPGit;

import org.apache.commons.math3.util.FastMath;

import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import moa.classifiers.core.driftdetection.HDDM_A_Test;
import moa.classifiers.core.driftdetection.HDDM_W_Test;
import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;

public class adaptiveHDDM_W extends HDDM_W_Test implements AdaptiveDetector {
	
	double adaptRange = 2; //degree to adapt conf interval, 1 is no adaptation, 2 is up to half/double of conf
	double origConf = -1;
	double adaptiveConf;
	
	public void setAdaptRange(double newRange){
		adaptRange = newRange;
	}
	
    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        resetLearning();
        adaptiveConf = this.driftConfidenceOption.getValue();
    }
	
	public void adaptConf(double adaptation){
		//adaptation is prob of drift between 0-1 from drift predictor
		if(adaptation < 0.5){

			this.driftConfidence = this.driftConfidenceOption.getValue() * (1/adaptRange + (2 * adaptation * (adaptRange-1)/adaptRange));
		}	
		else {
			this.driftConfidence = this.driftConfidenceOption.getValue() * (1 + (2 * (adaptation-0.5)*(adaptRange-1))); 

		}
	}
}
	