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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles=75;
	weights.resize(num_particles);
	particles.resize(num_particles);
	normal_distribution<double>dist_x(x,std[0]);
	normal_distribution<double>dist_y(y,std[1]);
	normal_distribution<double>dist_theta(theta,std[2]);
	for(int i=0;i<num_particles;i++){
		particles[i].id=i;
		particles[i].weight=1.0;
		particles[i].x=dist_x(gen);
		particles[i].y=dist_y(gen);
		particles[i].theta=dist_theta(gen);
		
	}
	is_initialized=true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double>dist_x(0,std_pos[0]);
	normal_distribution<double>dist_y(0,std_pos[1]);
	normal_distribution<double>dist_theta(0,std_pos[2]);
	for(int i=0;i<num_particles;i++){
		if(fabs(yaw_rate)>0.0001){
			particles[i].x+=velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y+=velocity/yaw_rate*(cos(particles[i].theta) -cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta+=yaw_rate*delta_t;
		}
		else{
			particles[i].x+=velocity*delta_t*cos(particles[i].theta);
			particles[i].y+=velocity*delta_t*sin(particles[i].theta);
			
		}
		particles[i].x+=dist_x(gen);
		particles[i].y+=dist_y(gen);
		particles[i].theta+=dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0;i<observations.size();i++){
		double distance_min=std::numeric_limits<double>::max();
		int min_index=-1;
		for(int j=0;j<predicted.size();j++){
			double distance=dist(observations[i].x,observations[i].y,predicted[j].x, predicted[j].y);
			if(distance<distance_min){
				distance_min=distance;
				min_index=j;
			}
		}
		observations[i].id=predicted[min_index].id;
		
	}



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	

	
	for(int i=0;i<num_particles;i++){
		std::vector<LandmarkObs> predictions;
		std::vector<LandmarkObs> observations_map_co_ord;
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		double cos_theta=cos(particles[i].theta);
		double sin_theta=sin(particles[i].theta);
		for(int k=0;k<observations.size();k++){
			LandmarkObs landmark;
			landmark.id=observations[k].id;
			landmark.x=particles[i].x+(cos_theta*observations[k].x)-(sin_theta*observations[k].y);
			landmark.y=particles[i].y+(sin_theta*observations[k].x)+(cos_theta*observations[k].y);
			observations_map_co_ord.push_back(landmark);
		}
		for(int j=0;j<map_landmarks.landmark_list.size();j++){
			if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) <= sensor_range){
				LandmarkObs landmark;
				landmark.id=map_landmarks.landmark_list[j].id_i;
				landmark.x=map_landmarks.landmark_list[j].x_f;
				landmark.y=map_landmarks.landmark_list[j].y_f;
				predictions.push_back(landmark);
			}
		}
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
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> resampled_particles(num_particles);
	default_random_engine gen;
	discrete_distribution<int> index(weights.begin(), weights.end());
	for(int i=0;i<num_particles;i++){
		
		resampled_particles[i]=particles[index(gen)];
	}
	particles=resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	// particle.associations.clear();
	// particle.sense_x.clear();
	// particle.sense_y.clear();

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
