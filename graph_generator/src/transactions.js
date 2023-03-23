import { serviceAccorderie } from "./data.js";
import { getRandomUser } from './members.js'
import { randomizer , formatDate, timeRandomizer} from "./utils.js";
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

  let acheteur;
  let vendeur;

  let done = false;
  while (!done) {
    acheteur = getRandomUser(users_list, "sociability_out");
    vendeur = getRandomUser(users_list, "sociability_in");
    if (acheteur.nom != vendeur.nom) {
      done = true;
    }
  }

  return [acheteur, vendeur];
}

/**
  Generates a random date in the desired interval.
  If biasfactor = 1, should return a date uniformly.  If > 1, favors dates closer to startdate, and if < 1, favors dates closer to enddate
  **/
export function generateTransactionDate(startdate, enddate, biasfactor = 1) {
  //convert to milliseconds, choose randomly millis within the time interval, convert to date, and return
  let mintime = startdate.getTime();
  let maxtime = enddate.getTime();

  let time =
    mintime + (maxtime - mintime) * Math.pow(Math.random(), biasfactor);

  return new Date(time);
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
    //const vendeur = generateRandomVendeur(users) || {};
    //const acheteur = generateRandomAcheteur(users, vendeur) || {};

    const transaction_date = generateTransactionDate(
      startdate,
      enddate,
      date_bias_factor
    );

    const acheteur_vendeur = generateTransactionSourceAndTarget(
      users,
      region_bias
    );
    const acheteur = acheteur_vendeur[0];
    const vendeur = acheteur_vendeur[1];

    const transaction = {
      service:
        serviceAccorderie[(0, randomizer(0, serviceAccorderie.length - 1))],
      vendeur: vendeur.nom,
      acheteur: acheteur.nom,
      date: formatDate(transaction_date),
      weight: timeRandomizer(),
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
    let key = u.nom;
    users_map[key] = u;
  }

  let detailed_transactions = [];

  for (const t of transactions) {
    const detailed_t = {
      service: t.service,
      vendeur: t.vendeur,
      acheteur: t.acheteur,
      date: t.date,
      weight: t.weight,
      longitude1: users_map[t.vendeur].longitude,
      latitude1: users_map[t.vendeur].latitude,
      longitude2: users_map[t.acheteur].longitude,
      latitude2: users_map[t.acheteur].latitude,
    };

    detailed_transactions.push(detailed_t);
  }

  return detailed_transactions;
}
