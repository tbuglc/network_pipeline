import { faker } from "@faker-js/faker";
import moment from "moment";
import {
  fakePostCode,
  getRandomRegionInfo,
  dateRandomizer,
  getRandomNumberInInterval,
  getRandomSociability,
} from "./utils.js";
/**
  Generate an array of users of size "size".  Each individual is assigned "sociability_out" and "sociability_in" weights, which 
  respectively control the probability that it is chosen as the tail of an edge, or the head of an edge
  size is the number of desired users 
  sociability_distribution can be "exp" for exponential distribution with lambda = sociability_params, or any other string to use uniform distribution
**/

export function generateUsers(
  size,
  sociability_distribution = "exp",
  sociability_params = 1.5
) {
  const users = [];
  if (!size) return users;

  for (let index = 0; index < size; index++) {
    const regionInfo = getRandomRegionInfo();
    const user = {
      nom: index,
      //TODO: Adjust user age from here
      age: dateRandomizer(moment("1987-01-01"), moment("1940-01-01")),
      address: fakePostCode(),
      region: regionInfo[0],
      longitude: regionInfo[1],
      latitude: regionInfo[2],
      genre: faker.name.sex(true) === 'female' ? 0 : 1,
      sociability_out: getRandomSociability(
        sociability_distribution,
        sociability_params
      ),
      sociability_in: getRandomSociability(
        sociability_distribution,
        sociability_params
      ),
    };

    users.push(user);
  }

  return users;
}

/**
  Choose a user based on their weights.  weight_attribute is the name of the user attribute 
  to use as a weight (intended to be either 'sociability_out' or 'sociability_in').
**/
export function getRandomUser(users, weight_attribute) {
  //the idea is to choose a random number r between 0 and the sum of weights.
  //we then go through the users and the first one whose sum reaches r is chosen.
  //todo: explain better
  //todo: find a better sampling strategy
  let attr_sum = 0;
  for (let i = 0; i < users.length; i++) {
    attr_sum += users[i][weight_attribute];
  }

  let r = getRandomNumberInInterval(0, attr_sum - 1);

  let tmp_sum = 0;
  for (let i = 0; i < users.length; i++) {
    let w = users[i][weight_attribute];

    if (tmp_sum + w > r) return users[i];

    tmp_sum += w;
  }
}
