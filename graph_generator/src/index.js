import { transactionDataColumn, userDataColumn } from "./data.js";

import { excelGenerator } from "./utils.js";
import { generateUsers } from "./members.js";
import {
  generateTransactions,
  generateDetailedTransactions,
} from "./transactions.js";

async function main(
  userSize,
  transactionSize,
  sociability_distribution = "exp",
  sociability_params = 1.5,
  region_bias = 0.6,
  date_bias_factor = 0.6,
  output_dir
) {
  const users = generateUsers(
    userSize,
    sociability_distribution,
    sociability_params
  );

  const startdate = new Date("2006-01-01");
  const enddate = new Date("2022-12-31");

  const transactions = generateTransactions(
    users,
    transactionSize,
    startdate,
    enddate,
    region_bias,
    date_bias_factor
  );

  await excelGenerator(users, "members", userDataColumn, output_dir);
  await excelGenerator(transactions, "transactions", transactionDataColumn, output_dir);

  //ML: add detailed transactions export
  // const detailed_transactionDataColumn = [
  //   ...transactionDataColumn,
  //   { key: "longitude1", header: "Longitude1" },
  //   { key: "latitude1", header: "Latitude1" },
  //   { key: "longitude2", header: "Longitude2" },
  //   { key: "latitude2", header: "Latitude2" },
  // ];

  // const detailed_transactions = generateDetailedTransactions(
  //   users,
  //   transactions
  // );
  // await excelGenerator(
  //   detailed_transactions,
  //   "detailed_transactions",
  //   detailed_transactionDataColumn,
  //   output_dir
  // );
}

/*process.argv.forEach(function (val, index, array) {
  console.log(index + ': ' + val);
});*/

var nbusers = 100;
var nbtransactions = 500;
var social_distrib = "exp";
var social_param = 1.5;
var region_bias = 0.5;
var date_bias_factor = 0.6;
var output_dir = './'
var doHelp = false;

// console.log(process.argv)
for (var i = 2; i < process.argv.length; i += 2) {

  if (process.argv[i] == "-o") output_dir = process.argv[i + 1];
  if (process.argv[i] == "-u") nbusers = process.argv[i + 1];
  if (process.argv[i] == "-t") nbtransactions = process.argv[i + 1];
  if (process.argv[i] == "-sd") social_distrib = process.argv[i + 1]; 
  if (process.argv[i] == "-sp") social_param = process.argv[i + 1];
  if (process.argv[i] == "-r") region_bias = process.argv[i + 1];
  if (process.argv[i] == "-d") date_bias_factor = process.argv[i + 1];
  if (process.argv[i] == "-h") {
    doHelp = true;
    break;
  }
}

if (doHelp) {
  const msg =
    "Destination folder \n" +
    "Arguments: \n" +
    "-o [nb] \n" +
    output_dir +
    " \n" +
    "Generates users and transactions \n" +
    "Arguments: \n" +
    "-u [nb] \n" +
    "Specifies that the number of users generated should be [nb], default=" +
    nbusers +
    " \n" +
    "-t [nb] \n" +
    "Specifies that the number of transactions generated should be [nb], default=" +
    nbtransactions +
    " \n" +
    "-r [p] \n" +
    "Region bias.  [p] is between 0 and 1 and is the probability that a transaction is forced to be in the same region.  Default=" +
    region_bias +
    " \n" +
    "-d [p] \n" +
    "Date bias factor.  [p] can be any number.  If biasfactor = 1, should return a date uniformly.  If > 1, favors dates closer to startdate, and if < 1, favors dates closer to enddate.  Default=" +
    date_bias_factor +
    " \n" +
    "-sd [str] \n" +
    "Sociability distribution for users.  Each individual is assigned 'sociability_out' and 'sociability_in' durees, which respectively control the probability that it is chosen as the tail of an edge, or the head of an edge.  str can be 'exp' for exponential distribution (see wiki), or any other string to use uniform distribution.  Under uniform, all nodes have the same number of neighbors on average, and under 'exp', some nodes will have more interactions than others.  Default=" +
    social_distrib +
    " \n" +
    "-sp [p] \n" +
    "Lambda parameter to use if exponential distribution is chosen.  Default=" +
    social_param;
  console.log(msg);
} else {
  // console.log(
  //   "Generating " +
  //     nbusers.toString() +
  //     " users and " +
  //     nbtransactions.toString() +
  //     " transactions"
  // );
  // console.log(
  //   "socialibility_distribution = " +
  //     social_distrib +
  //     "(" +
  //     social_param +
  //     "), region_bias = " +
  //     region_bias +
  //     ", date_bias = " +
  //     date_bias_factor
  // );

  main(
    nbusers,
    nbtransactions,
    social_distrib,
    social_param,
    region_bias,
    date_bias_factor,
    output_dir
  );
}
