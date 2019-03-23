package io.github.dominikthomas.neuralnet.math;

import java.util.Random;

/**
 * 
 * RandomNumberGenerator
 * This class generates double precision random numbers according to a seed. It 
 * is used in weights initialization, for example.
 * 
 * @author Alan de Souza, Fábio Soares, Dominik Thomas
 * @version 0.2
 */
public class RandomNumberGenerator {
    /**
     * Seed that is used for random number generation
     */
    public static long seed=System.currentTimeMillis();
    /**
     * Random singleton object that actually generates the random numbers
     */
    private static Random r = new Random(seed);
    /**
     * GenerateNext
     * Static method that returns a newly random number
     * @return 
     */
    public static double GenerateNext(){
        if(r==null)
            r = new Random(seed);
        return r.nextDouble();
    }
    
    /** 
     * setSeed
     * Sets a new seed for the random generator
     * @param seed new seed for random generator
     */
    public static void setSeed(long newSeed){
        seed=newSeed;
        r.setSeed(newSeed);
    }

    public static double GenerateBetween(double min,double max){
        if(r==null)
            r=new Random(seed);
        if(max<min)
           return min;
        return min+(r.nextDouble()*(max-min));
    }
}
