package io.github.dominikthomas.neuralnet.data;

import io.github.dominikthomas.neuralnet.beans.NeuralNet;
import java.util.ArrayList;

public class NeuralDataSet {
		public NeuralInputData inputData;
	    public NeuralOutputData outputData;
	    
	    public NeuralNet neuralNet;
	    
	    public int numberOfInputs;
	    public int numberOfOutputs;
	    
	    public int numberOfRecords;
	    
	    public NeuralDataSet() {
	    }
	    
	    public void init() {
		    numberOfInputs=inputData.numberOfInputs;
		    numberOfOutputs=outputData.numberOfOutputs;
		    numberOfRecords=inputData.data.size();
		    
	    }
	    
	    public ArrayList<ArrayList<Double>> getArrayInputData(){
	        return inputData.data;
	    }
	    
	    public ArrayList<ArrayList<Double>> getArrayTargetOutputData(){
	        return outputData.getTargetDataArrayList();
	    }
	    
	    public ArrayList<ArrayList<Double>> getArrayNeuralOutputData(){
	        return outputData.getNeuralDataArrayList();
	    }
	    
	    public ArrayList<Double> getArrayInputRecord(int i){
	        return inputData.getRecordArrayList(i);
	    }
	    
	    public double[] getInputRecord(int i){
	        return inputData.getRecord(i);
	    }
	    
	    public ArrayList<Double> getArrayTargetOutputRecord(int i){
	        return outputData.getTargetRecordArrayList(i);
	    }
	    
	    public double[] getTargetOutputRecord(int i){
	        return outputData.getTargetRecord(i);
	    }
	    
	    public ArrayList<Double> getArrayNeuralOutputRecord(int i){
	        return outputData.getRecordArrayList(i);
	    }
	    
	    public double[] getNeuralOutputRecord(int i){
	        return outputData.getRecord(i);
	    }
	    
	    public void setNeuralOutput(int i,ArrayList<Double> _neuralData){
	        this.outputData.setNeuralData(i, _neuralData);
	    }
	    
	    public void setNeuralOutput(int i,double[] _neuralData){
	        this.outputData.setNeuralData(i, _neuralData);
	    }
	    
	    public ArrayList<Double> getIthInputArrayList(int i){
	        return this.inputData.getColumnDataArrayList(i);
	    }
	    
	    public double[] getIthInput(int i){
	        return this.inputData.getColumn(i);
	    }
	    
	    public ArrayList<Double> getIthTargetOutputArrayList(int i){
	        return this.outputData.getTargetColumnArrayList(i);
	    }
	    
	    public double[] getIthTargetOutput(int i){
	        return this.outputData.getTargetColumn(i);
	    }
	    
	    public ArrayList<Double> getIthNeuralOutputArrayList(int i){
	        return this.outputData.getNeuralColumnArrayList(i);
	    }
	    
	    public double[] getIthNeuralOutput(int i){
	        return this.outputData.getNeuralColumn(i);
	    }
	    
	    public void printInput(){
	        this.inputData.print();
	    }
	    
	    public void printTargetOutput(){
	        this.outputData.printTarget();
	    }
	    
	    public void printNeuralOutput(){
	        this.outputData.printNeural();
	    }
	    
	    public ArrayList<Double> getMeanInput(){
	        return this.inputData.getMeanInputData();
	    }
	    
	    public ArrayList<Double> getMeanNeuralOutput(){
	        return this.outputData.getMeanNeuralData();
	    }
	    
	    public void setInputData(NeuralInputData inputData) {
			this.inputData = inputData;
		}
	    
	    public void setOutputData(NeuralOutputData outputData) {
			this.outputData = outputData;
		}
	    
	}
