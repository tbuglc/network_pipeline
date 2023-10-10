import { serviceAccorderie } from "./data.js";
import { getRandomUser } from './members.js'
import { randomizer, formatDate, timeRandomizer } from "./utils.js";
import { regions } from "./constants.js";

/**
  Generate source and target of a transaction, returns them [source, target]
  region_bias determines the probability that we force a transaction in the same region.
  Close to 0 = no bias, close to 1 = all transactions in the same region
**/

export function generateTransactionSourceAndTarget(users, region_bias) {
  const r = Math.random();

  let users_list;

  if (r < region_bias) {
    //forced region => users_list is restricted to users in region
    let region_info = regions[randomizer(0, regions.length - 1)];

    users_list = [];
    for (var i = 0; i < users.length; i++) {
      if (users[i].region == region_info[0]) users_list.push(users[i]);
    }

    //hopefully, some region has at least two people, otherwise this loops infinitely
    if (users_list.length <= 1) {
      return generateTransactionSourceAndTarget(users, region_bias);
    }
  } else {
    //unforced region => all user pairs are possible
    users_list = users;
  }

  let target;
  let source;

  let done = false;
  while (!done) {
    target = getRandomUser(users_list, "sociability_out");
    source = getRandomUser(users_list, "sociability_in");
    if (target.id != source.id) {
      done = true;
    }
  }

  return [target, source];
}
// Function to generate a random date with a user-defined bias factor
function generateRandomDateWithBias(startDate, endDate, userBiasFactor) {
  // console.log(startDate, endDate, userBiasFactor)
  // If userBiasFactor is 0, return a completely random date
  if (userBiasFactor === 0) {
    const randomTimestamp = startDate + Math.random() * (endDate - startDate);
    return new Date(randomTimestamp);
  }

  // Define the range of biasFactor (-1 to 1)
  const minBiasFactor = -1; // Bias towards startDate
  const maxBiasFactor = 1;  // Bias towards endDate

  // Ensure userBiasFactor is within the valid range (-1 to 1)
  const biasFactor = Math.min(maxBiasFactor, Math.max(minBiasFactor, userBiasFactor));

  // Generate a random number between 0 and 1
  const random = Math.random();

  // Apply the bias to the random number
  const biasedRandom = 1 - Math.pow(random, Math.abs(biasFactor));

  // Calculate the random timestamp biased towards startDate or endDate
  const randomTimestamp = startDate + (biasedRandom * (endDate - startDate));
  // console.log(randomTimestamp)
  // Create a new Date object from the random timestamp
  const randomDate = new Date(randomTimestamp);
  // console.log(randomDate)
  return randomDate;
}

/**
  Generates a random date in the desired interval.
  If biasfactor = 1, should return a date uniformly.  If > 1, favors dates closer to startdate, and if < 1, favors dates closer to enddate
  **/
export function generateTransactionDate(startdate, enddate, biasfactor = 0) {
  // console.log(startdate, enddate)
  //convert to milliseconds, choose randomly millis within the time interval, convert to date, and return
  // let mintime = startdate.getTime();
  // let maxtime = enddate.getTime();

  // let time = mintime + (maxtime - mintimet) * Math.pow(Math.random(), biasfactor);

  // return new Date(time);
  const result = generateRandomDateWithBias(startdate.getTime(), enddate.getTime(), biasfactor)
  // console.log('result', result)
  return result
}

/**
  region_bias is the probability that a transaction is forced to be inside the same region. 
  date_bias_factor determines the probability that the date is closer to end date.  See generateTransactionDate
**/
export function generateTransactions(
  users,
  transactionSize,
  startdate,
  enddate,
  region_bias = 0,
  date_bias_factor = 0.6
) {
  const transactions = [];
  for (let index = 0; index < transactionSize; index++) {
    //const source = generateRandomsource(users) || {};
    //const target = generateRandomtarget(users, source) || {};

    const transaction_date = generateTransactionDate(
      startdate,
      enddate,
      date_bias_factor
    );

    const target_source = generateTransactionSourceAndTarget(
      users,
      region_bias
    );
    const target = target_source[0];
    const source = target_source[1];

    const transaction = {
      service:
        serviceAccorderie[(0, randomizer(0, serviceAccorderie.length - 1))],
      source: source.id,
      target: target.id,
      accorderie: 109,
      id: index,
      date: formatDate(transaction_date),
      duree: timeRandomizer(),
    };

    transactions.push(transaction);
  }

  return transactions;
}

/**
  generate detailed transactions file, where each line has all the info about both users
  **/
export function generateDetailedTransactions(users, transactions) {
  let users_map = {};
  for (const u of users) {
    let key = u.id;
    users_map[key] = u;
  }

  let detailed_transactions = [];

  for (const t of transactions) {
    const detailed_t = {
      service: t.service,
      source: t.source,
      target: t.target,
      date: t.date,
      duree: t.duree,
      longitude1: users_map[t.source].longitude,
      latitude1: users_map[t.source].latitude,
      longitude2: users_map[t.target].longitude,
      latitude2: users_map[t.target].latitude,
    };

    detailed_transactions.push(detailed_t);
  }

  return detailed_transactions;
}
