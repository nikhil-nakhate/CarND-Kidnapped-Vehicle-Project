/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits.h>

#include "helper_functions.h"

#define EPS 0.00001

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

    if(is_initialized) {
        return;
    }

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 100;  // TODO: Set the number of particles
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i = 0; i < num_particles; i++) {
        if(fabs(yaw_rate) < EPS) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else{
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    for(auto &o : observations) {
        double min_dist = INT_MAX;
        int min_idx = -1;
        for(auto &p : predicted) {
            double curr_dist = dist(p.x, p.y, o.x, o.y);
            if(curr_dist < min_dist) {
                min_dist = curr_dist;
                min_idx = p.id;
            }
        }
        o.id = min_idx;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    for(auto& p : particles) {

        vector<LandmarkObs> lmks_in_range;
        for(auto l : map_landmarks.landmark_list) {
            if(dist(l.x_f, l.y_f, p.x, p.y) <= sensor_range) {
                lmks_in_range.push_back(LandmarkObs{ l.id_i, l.x_f, l.y_f });
            }
        }

        vector<LandmarkObs> trans_obs;
        for(auto o : observations) {
            std::pair<double, double> map_obs = transform_obs(p.x, p.y, p.theta, o.x, o.y);
            trans_obs.push_back(LandmarkObs{o.id, map_obs.first, map_obs.second});
        }

        dataAssociation(lmks_in_range, trans_obs);

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        double wei;
        p.weight = 1.0;
        for(auto ob: trans_obs) {
            double l_x=0.0, l_y = 0.0;
            int asso_prediction = ob.id;

            for(auto& lmk : lmks_in_range) {
                if(lmk.id == asso_prediction) {
                    l_x = lmk.x;
                    l_y = lmk.y;
                }
            }
            wei = multi_var_gauss(ob.x, ob.y, l_x, l_y, std_landmark[0], std_landmark[1]);
            if (wei == 0) {
                p.weight *= EPS;
            }else {
                p.weight *= wei;
            }
            associations.push_back(asso_prediction);
            sense_x.push_back(ob.x);
            sense_y.push_back(ob.y);
        }
        SetAssociations(p, associations, sense_x, sense_y);
    }

}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    weights.clear();
    for(int i=0; i<num_particles; ++i){
        weights.push_back(particles[i].weight);
    }

    std::discrete_distribution<int> particle_dist(weights.begin(),weights.end());

    // Resample particles
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for(int i=0; i<num_particles; ++i){
        auto index = particle_dist(gen);
        new_particles[i] = std::move(particles[index]);
    }
    particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}