/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package listeners;

import org.jfree.data.xy.XYSeries;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;

/**
 *
 * @author douglas
 */
public class NeuralNetworkLearningEventListener implements LearningEventListener{

    // Gráfico da época (X) pelo erro quadrático (Y)
    public XYSeries erros = new XYSeries("Erro Quadrático Médio");
    public int epocaMenorErro;
    public double menorErro = 10;
    public Double[] pesosMenorErro;
    
    @Override
    public void handleLearningEvent(LearningEvent le) {
    
        LearningRule lr = (LearningRule) le.getSource();
        MultiLayerPerceptron mlp = (MultiLayerPerceptron) lr.getNeuralNetwork();

        double erro = mlp.getLearningRule().getErrorFunction().getTotalError();
        int epoca = mlp.getLearningRule().getCurrentIteration();

        erros.add(epoca, erro);
        
        if(erro < menorErro) {
            menorErro = erro;
            pesosMenorErro = mlp.getWeights();
        }
        
        //nne.getEventType().toString();
        
    }
    
}
