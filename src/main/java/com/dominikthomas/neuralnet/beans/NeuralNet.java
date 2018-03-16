package com.dominikthomas.neuralnet.beans;

import com.dominikthomas.neuralnet.math.IActivationFunction;
import java.util.ArrayList;

/**
 *
 * NeuralNet
 * This class represents the Neural Network itself. It contains all the 
 * definitions that a Neural Network has, including method for calculation 
 * (forward).
 * 
 * @author Alan de Souza, Fábio Soares, Dominik Thomas
 * @version 0.2
 */
public class NeuralNet {
    
    /**
     * Neural Network Input Layer
     */
    private InputLayer inputLayer;
    /**
     * Neural Network array of hidden layers, that may contain 0 or many
     */
    private ArrayList<HiddenLayer> hiddenLayer;
    /**
     * Neural Network Output Layer
     */
    private OutputLayer outputLayer;
    /**
     * List of number of neurons per hidden Layer
     */
    private ArrayList<Integer> numberOfHiddenNeurons;
    /**
     * List of Activation functions per hidden Layer
     */
    private ArrayList<IActivationFunction> hiddenAcFnc;
    /**
     * Output Activation function
     */
    private IActivationFunction outputAcFnc;
    /**
     * Number of Hidden Layers
     */
    private int numberOfHiddenLayers;
    /**
     * Number of Inputs
     */
    private int numberOfInputs;
    /**
     * Number of Outputs
     */
    private int numberOfOutputs;
    
    /**
     * Array of neural inputs
     */
    private ArrayList<Double> input;
    /**
     * Array of neural outputs
     */
    private ArrayList<Double> output;
    
    /**
     * NeuralNet constructor
     * This constructor initializes the neural network by initializing all of 
     * the underlying layers and their respective neurons.
     * 
     * @param numberofinputs Number of Inputs of this Neural Network
     * @param numberofoutputs Number of Outputs of this Neural Network
     * @param numberofhiddenneurons Array containing the number of Neurons in 
     * each of the Hidden Layers
     * @param hiddenAcFnc Array containing the activation function of each 
     * Hidden Layer
     * @param outputAcFnc Activation Function of the Output Layer
     */
    public NeuralNet(){
    }
    
    public void init() throws Exception{
    
        numberOfHiddenLayers=numberOfHiddenNeurons.size();
        if(numberOfHiddenLayers==hiddenAcFnc.size()){
            input=new ArrayList<>(numberOfInputs);
            inputLayer=new InputLayer(numberOfInputs);
            if(numberOfHiddenLayers>0){
                hiddenLayer=new ArrayList<>(numberOfHiddenLayers);
            }
            for(int i=0;i<numberOfHiddenLayers;i++){
                if(i==0){
                    try{
                        hiddenLayer.set(i,new HiddenLayer(numberOfHiddenNeurons.get(i),
                            hiddenAcFnc.get(i),
                            inputLayer.getNumberOfNeuronsInLayer()));
                    }
                    catch(IndexOutOfBoundsException iobe){
                        hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons.get(i),
                            hiddenAcFnc.get(i),
                            inputLayer.getNumberOfNeuronsInLayer()));
                    }
                    inputLayer.setNextLayer(hiddenLayer.get(i));
                }
                else{
                    try{
                        hiddenLayer.set(i, new HiddenLayer(numberOfHiddenNeurons.get(i),
                             hiddenAcFnc.get(i),hiddenLayer.get(i-1)
                            .getNumberOfNeuronsInLayer()
                            ));
                    }
                    catch(IndexOutOfBoundsException iobe){
                        hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons.get(i),
                             hiddenAcFnc.get(i),hiddenLayer.get(i-1)
                            .getNumberOfNeuronsInLayer()
                            ));
                    }
                    hiddenLayer.get(i-1).setNextLayer(hiddenLayer.get(i));
                }
            }
            if(numberOfHiddenLayers>0){
                outputLayer=new OutputLayer(numberOfOutputs,outputAcFnc,
                        hiddenLayer.get(numberOfHiddenLayers-1)
                        .getNumberOfNeuronsInLayer() 
                        );
                hiddenLayer.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);
            }
            else{
                outputLayer=new OutputLayer(numberOfInputs, outputAcFnc,
                		numberOfOutputs);
                inputLayer.setNextLayer(outputLayer);
            }
        } else {
        	throw new Exception ("Number of hidden layers does not equal the number of activation functions.");
        }
    }
    
    /**
     * setInputs
     * Feeds an array of real values to the neural network's inputs
     * @param inputs Array of real values to be fed into the neural inputs
     */
    public void setInputs(ArrayList<Double> inputs){
        if(inputs.size()==numberOfInputs){
            this.input=inputs;
        }
    }
    
    /**
     * setInputs
     * Sets a vector of double-precision values into the neural network inputs
     * @param inputs vector of values to be fed into the neural inputs
     */
    public void setInputs(double[] inputs){
        if(inputs.length==numberOfInputs){
            for(int i=0;i<numberOfInputs;i++){
                try{
                    input.set(i, inputs[i]);
                }
                catch(IndexOutOfBoundsException iobe){
                    input.add(inputs[i]);
                }
            }
        }
    }
    
    /**
     * calc
     * This method calculates the output of each layer and forwards all values 
     * to the next layer
     */
    public void calc(){
        inputLayer.setInputs(input);
        inputLayer.calc();
        if(numberOfHiddenLayers>0){
            for(int i=0;i<numberOfHiddenLayers;i++){
                HiddenLayer hl = hiddenLayer.get(i);
                hl.setInputs(hl.getPreviousLayer().getOutputs());
                hl.calc();
            }
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        this.output=outputLayer.getOutputs();
    }
    
    /**
     * getArrayOutputs
     * @return Returns the neural outputs in the form of Array
     */
    public ArrayList<Double> getArrayOutputs(){
        return output;
    }
    
    /**
     * getOutputs
     * @return Returns the neural outputs in the form of vector
     */
    public double[] getOutputs(){
        double[] _outputs = new double[numberOfOutputs];
        for(int i=0;i<numberOfOutputs;i++){
            _outputs[i]=output.get(i);
        }
        return _outputs;
    }
    
    public void setNumberOfInputs(int numberOfInputs) {
		this.numberOfInputs = numberOfInputs;
	}
    
    public void setNumberOfOutputs(int numberOfOutputs) {
		this.numberOfOutputs = numberOfOutputs;
	}
    
    public void setNumberOfHiddenNeurons(ArrayList<Integer> numberOfHiddenNeurons) {
		this.numberOfHiddenNeurons = numberOfHiddenNeurons;
	}
    public void setHiddenAcFnc(ArrayList<IActivationFunction> hiddenAcFnc) {
		this.hiddenAcFnc = hiddenAcFnc;
	}
    public void setOutputAcFnc(IActivationFunction outputAcFnc) {
		this.outputAcFnc = outputAcFnc;
	}
}
