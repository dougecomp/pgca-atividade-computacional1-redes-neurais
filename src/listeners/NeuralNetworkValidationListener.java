/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package listeners;

import org.jfree.data.xy.XYSeries;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.events.NeuralNetworkEvent;
import org.neuroph.nnet.MultiLayerPerceptron;

/**
 *
 * @author douglas
 */
public class NeuralNetworkValidationListener implements LearningEventListener {

    // Gráfico da época (X) pelo erro quadrático (Y)
    public XYSeries erros = new XYSeries("Erro da Rede");

    @Override
    public void handleLearningEvent(LearningEvent le) {
        
    }
    
}
