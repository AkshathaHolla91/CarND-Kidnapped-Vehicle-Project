/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	default_random_engine gen;
	//Set the number of particles
	num_particles=75;
	weights.resize(num_particles);
	particles.resize(num_particles);
	//Adding random guassian noise to each particle
	normal_distribution<double>dist_x(x,std[0]);
	normal_distribution<double>dist_y(y,std[1]);
	normal_distribution<double>dist_theta(theta,std[2]);
	for(int i=0;i<num_particles;i++){
		particles[i].id=i;
		//Initializing the weight of the particle to 1
		particles[i].weight=1.0;
		//Set initial position of particles randomly from a guassian of GPS x,y and theta
		particles[i].x=dist_x(gen);
		particles[i].y=dist_y(gen);
		particles[i].theta=dist_theta(gen);
		
	}
	is_initialized=true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	default_random_engine gen;
	//Random guassian noise generators 
	normal_distribution<double>dist_x(0,std_pos[0]);
	normal_distribution<double>dist_y(0,std_pos[1]);
	normal_distribution<double>dist_theta(0,std_pos[2]);
	for(int i=0;i<num_particles;i++){
		//Set the values of x, y and theta of each particle with the respective equations based on whether yaw rate is 0 or not.
		if(fabs(yaw_rate)>0.0001){
			particles[i].x+=velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y+=velocity/yaw_rate*(cos(particles[i].theta) -cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta+=yaw_rate*delta_t;
		}
		else{
			particles[i].x+=velocity*delta_t*cos(particles[i].theta);
			particles[i].y+=velocity*delta_t*sin(particles[i].theta);
			
		}
		//Adding random noise to x , y and theta
		particles[i].x+=dist_x(gen);
		particles[i].y+=dist_y(gen);
		particles[i].theta+=dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
	
	for(int i=0;i<observations.size();i++){
		//Initializing min distance to max value of double
		double distance_min=std::numeric_limits<double>::max();
		//Initilaizing min index to -1
		int min_index=-1;
		for(int j=0;j<predicted.size();j++){
			//Calculate euclidian distance between the predicted measurement and  observation
			double distance=dist(observations[i].x,observations[i].y,predicted[j].x, predicted[j].y);
			//Update the distance_min and min index with the minimum values
			if(distance<distance_min){
				distance_min=distance;
				min_index=j;
			}
		}
		//Set the id of the closest predicted measurement as the id of the particular observation
		observations[i].id=predicted[min_index].id;
		
	}



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
		
	for(int i=0;i<num_particles;i++){
		std::vector<LandmarkObs> predictions;
		std::vector<LandmarkObs> observations_map_co_ord;
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		double cos_theta=cos(particles[i].theta);
		double sin_theta=sin(particles[i].theta);
		//Convert the observations from car co-ordinate system to map co-ordinates
		for(int k=0;k<observations.size();k++){
			LandmarkObs landmark;
			landmark.id=observations[k].id;
			landmark.x=particles[i].x+(cos_theta*observations[k].x)-(sin_theta*observations[k].y);
			landmark.y=particles[i].y+(sin_theta*observations[k].x)+(cos_theta*observations[k].y);
			observations_map_co_ord.push_back(landmark);
		}
		//Prepare list of predictions by calculating distance between each particle and various landmarks
		for(int j=0;j<map_landmarks.landmark_list.size();j++){
			if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) <= sensor_range){
				LandmarkObs landmark;
				landmark.id=map_landmarks.landmark_list[j].id_i;
				landmark.x=map_landmarks.landmark_list[j].x_f;
				landmark.y=map_landmarks.landmark_list[j].y_f;
				predictions.push_back(landmark);
			}
		}
		//Perform data association 
		dataAssociation(predictions, observations_map_co_ord);
		for(int l=0;l<observations_map_co_ord.size();l++){
			associations.push_back(observations_map_co_ord[l].id);
			sense_x.push_back(observations_map_co_ord[l].x);
			sense_y.push_back(observations_map_co_ord[l].y);
		}
		SetAssociations(particles[i], associations, sense_x, sense_y);
		double normalization_constant=1.0/(2.0*M_PI*std_landmark[0]*std_landmark[1]);
		double variance_x=2*std_landmark[0]*std_landmark[0];
		double variance_y=2*std_landmark[1]*std_landmark[1];
		double multi_variate_gaussian=1;
		particles[i].weight=1;
		//Calculate the multivariate guassian probability to find the measurement probability and get the final weight by multiplying all of them
		for(int m=0;m<observations_map_co_ord.size();m++){
			for(int n=0;n<predictions.size();n++){
				if(observations_map_co_ord[m].id==predictions[n].id){
					multi_variate_gaussian=normalization_constant*exp(-(pow(predictions[n].x-observations_map_co_ord[m].x,2)/variance_x+pow(predictions[n].y-observations_map_co_ord[m].y,2)/variance_y));
					particles[i].weight*=multi_variate_gaussian;
					weights[i]=particles[i].weight;
					break;
				}
			}
		}
	}



	
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	
	std::vector<Particle> resampled_particles(num_particles);
	default_random_engine gen;
	//Using discrete distribution to return particles by weight
	discrete_distribution<int> index(weights.begin(), weights.end());
	for(int i=0;i<num_particles;i++){
		
		resampled_particles[i]=particles[index(gen)];
	}
	//Assigning the resampled particles to the particle vector
	particles=resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
