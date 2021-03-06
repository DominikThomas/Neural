package io.github.dominikthomas.neuralnet.learn;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.beans.NeuralNet;
import io.github.dominikthomas.neuralnet.data.NeuralDataSet;

public abstract class LearningAlgorithm {
    
    protected NeuralNet neuralNet;
    
    public enum LearningMode {ONLINE,BATCH};
    
    protected enum LearningParadigm {SUPERVISED,UNSUPERVISED};
    
    protected LearningMode learningMode;
    
    protected LearningParadigm learningParadigm;
    
    protected int MaxEpochs=100;
    
    protected int epoch=0;
    
    protected double MinOverallError=0.001;
    
    protected double LearningRate=0.1;
    
    protected NeuralDataSet trainingDataSet;
    
    protected NeuralDataSet testingDataSet;
    
    protected NeuralDataSet validatingDataSet;
    
    public boolean printTraining=false;
    
    public abstract void train() throws NeuralException;
    
    public abstract void forward() throws NeuralException;
    
    public abstract void forward(int i) throws NeuralException;
    
    public abstract Double calcNewWeight(int layer,int input,int neuron) throws NeuralException;
    
    public abstract Double calcNewWeight(int layer,int input,int neuron,double error) throws NeuralException;
    
    public abstract void test() throws NeuralException;
    
    public abstract void test(int i) throws NeuralException;
    
    public abstract void print();
    
    public void setMaxEpochs(int _maxEpochs){
        this.MaxEpochs=_maxEpochs;
    }
    
    public int getMaxEpochs(){
        return this.MaxEpochs;
    }
    
    public int getEpoch(){
        return epoch;
    }
    
    public void setMinOverallError(Double _minOverallError){
        this.MinOverallError=_minOverallError;
    }
    
    public Double getMinOverallError(){
        return this.MinOverallError;
    }
    
    public void setLearningRate(Double _learningRate){
        this.LearningRate=_learningRate;
    }
    
    public Double getLearningDate(){
        return this.LearningRate;
    }
    
    public void setLearningMode(LearningMode _learningMode){
        this.learningMode=_learningMode;
    }
    
    public LearningMode getLearningMode(){
        return this.learningMode;
    }
    
    public void setPrintTraining(boolean printTraining) {
		this.printTraining = printTraining;
	}
    
    public void setNeuralNet(NeuralNet neuralNet) {
    	neuralNet.init();
		this.neuralNet = neuralNet;
	}
    
    public NeuralNet getNeuralNet() {
    	return neuralNet;
    }
    
    public void setTrainingDataSet(NeuralDataSet trainingDataSet) {
		this.trainingDataSet = trainingDataSet;
	}
    
    public NeuralDataSet getTrainingDataSet() {
		return trainingDataSet;
	}
    
    public void setTestingDataSet(NeuralDataSet _testingDataSet){
    	this.testingDataSet=_testingDataSet;
    }
    
    public NeuralDataSet getTestingDataSet() {
		return testingDataSet;
	}
    
}
