package io.github.dominikthomas.neuralnet.beans;

/**
 *
 * This class represents an output layer of a neural network, inheriting from 
 * NeuralLayer, which contains all basic definitions of a Neural Layer
 * 
 * @author Alan de Souza, Fabio Soares, Dominik Thomas
 * @version 0.2
 */
public class OutputLayer extends NeuralLayer {
    
    public OutputLayer() {
	}
    
    /**
     * This method prevents any attempt to link this layer to a next one, 
     * provided that this should be always the last
     * @param layer Dummy layer 
     */
    @Override
    public void setNextLayer(NeuralLayer layer){
        nextLayer=null;
    }
    
    /**
     * This method links this layer to the previous one
     * @param layer Previous Layer
     */
    @Override
    public void setPreviousLayer(NeuralLayer layer){
        previousLayer=layer;
        if(layer.nextLayer!=this)
            layer.setNextLayer(this);
    }
    
}
