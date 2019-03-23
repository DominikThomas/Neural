package com.dominikthomas.neuralnet.beans;

import com.dominikthomas.neuralnet.init.WeightInitialization;
import com.dominikthomas.neuralnet.math.Linear;
import java.util.ArrayList;

/**
 *
 * This class extends from NeuralLayer and represents the input layer of the 
 * Neural Network
 * 
 * @author Alan de Souza, Fabio Soares
 * @version 0.1
 */
public class InputLayer extends NeuralLayer {
    
    /**
     * InputLayer constructor
     * @param numberofinputs Number of Inputs of this layer and the Neural 
     * Network
     * @param _neuralNet 
     * @see InputLayer
     */
    public InputLayer(NeuralNet _neuralNet,int numberofinputs){
        super(_neuralNet,numberofinputs,new Linear(1));
        previousLayer=null;
        numberOfInputs=numberofinputs;
        this.init(_neuralNet.getWeightInitialization());
    }
    
    /**
     * This method links this layer to a next layer in the Neural Network
     * 
     * @param layer Next Neural Layer
     * @see InputLayer
     */
    @Override
    public void setNextLayer(NeuralLayer layer){
        nextLayer=layer;
        if(layer.previousLayer!=this)
            layer.setPreviousLayer(this);
    }
    
    /**
     * This method prevents any attempt to link this layer to a previous one, 
     * provided that this should be the first layer
     * 
     * @param layer dummy Neural Layer
     * @see InputLayer
     */
    @Override
    public void setPreviousLayer(NeuralLayer layer){
        previousLayer=null;
    }
    
    /**
     * This method initializes all neurons of this layer
     * 
     * @see InputLayer
     */
    @Override
    public void init(WeightInitialization weightInitialization){
        for(int i=0;i<numberOfInputs;i++){
            this.setNeuron(i,new InputNeuron());
            this.getNeuron(i).setNeuralLayer(this);
            this.getNeuron(i).init();
        }
    }
    
    /**
     * This method feeds an array of real values into this layer's inputs
     * 
     * @param inputs array of values to be fed into the layer's inputs
     * @see InputLayer
     */
    @Override
    public void setInputs(ArrayList<Double> inputs){
        if(inputs.size()==numberOfInputs){
            input=inputs;
        }
    }
    
    /**
     * This method overrides the superclass calc because it just passes the 
     * input values to the outputs, provided this is the input layer
     * 
     * @see InputLayer
     */
    @Override
    public void calc(){
        if(input!=null && getListOfNeurons()!=null){
            for(int i=0;i<numberOfNeuronsInLayer;i++){
                double[] firstInput = {this.input.get(i)};
                getNeuron(i).setInputs(firstInput);
                getNeuron(i).calc();
                try{
                    output.set(i,getNeuron(i).getOutput());
                }
                catch(IndexOutOfBoundsException iobe){
                    output.add(getNeuron(i).getOutput());
                }
            }
        }
    }
    
}
