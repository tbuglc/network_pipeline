import ExcelJS from "exceljs";
import { join } from "path";
import _ from "lodash";
import randomDate from "moment-random";
import { regions, ageRanges } from "./constants.js";
import { alpha } from "./data.js";
import moment from "moment";
import * as fs from 'fs'

//catégo plus haut niveau
//Celia

export function randomizer(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function dateRandomizer(end, start) {
  return randomDate(end, start).format("YYYY-MM-DD");
}

export function hourNumberFormat(number) {
  if (number < 9) return `0${number}`;
  return number;
}

export function timeRandomizer(start = 0, end = 6) {
  // return `${hourNumberFormat(randomizer(start, end))}:${hourNumberFormat(
  //   randomizer(0, 60)
  // )}`;

  return randomizer(start, end);
}

export function fakePostCode() {
  //FORMAT: ANA NAN
  return `${alpha[randomizer(0, 23)]}${randomizer(1, 9)}${
    alpha[randomizer(0, 23)]
  } ${randomizer(1, 9)}${alpha[randomizer(0, 23)]}${randomizer(1, 9)}`;
}

export function getRandomNumberInInterval(min, max) {
  return Math.random() * (max - min) + min;
}

/**
    Each region is represented as an array of 5 values.  
    TODO: one could argue that using objects would be better.
    Format is 
    [name, center_longitude, center_latitude, box_width, box_height]
    generated by hand from https://cartes.ville.sherbrooke.qc.ca/carteglobale/?theme=limitesetservices&cat=arrondissement
  **/

/**
    Takes a random region and returns a triple [r, a, b] where 
    r is the region name 
    a, b are random longitude/latitude in the region box
  **/
export function getRandomRegionInfo() {
  const random = Math.floor(Math.random() * regions.length);

  const r = regions[random];
  const delta_long = getRandomNumberInInterval(-1 * r[3], r[3]);
  const delta_lat = getRandomNumberInInterval(-1 * r[4], r[4]);

  return [r[0], r[1] + delta_long, r[2] + delta_lat];
}

/**
    Returns a number chosen from the exponential distribution.  rate is the lambda parameter (see wiki)
  **/
export function randomExponential(rate) {
  // http://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
  rate = rate || 1;
  // console.log(rate)
  var U = Math.random();
  // console.log(U)
  return -Math.log(U) / rate;
}

/**
    The sociability of an individual is an integer that durees the probability of being the endpoint of an edge
    distribution can either be "exp" for exponential, or anything else, which defaults to uniform 
    TODO: I am not sure about what the param controls
  **/
export function getRandomSociability(distribution = "exp", param = 1) {
  if (distribution == "exponential" || distribution == "exp") {
    return Math.ceil(randomExponential(param) + 0.0001);
  } else {
    return 1;
  }
}

export function formatDate(date) {
  var d = new Date(date),
    month = "" + (d.getMonth() + 1),
    day = "" + d.getDate(),
    year = d.getFullYear();

  if (month.length < 2) month = "0" + month;
  if (day.length < 2) day = "0" + day;

  return [year, month, day].join("-");
}
export async function excelGenerator(data, sheetLabel, columns, path) {
  const wb = new ExcelJS.Workbook();
  const ws = wb.addWorksheet(sheetLabel);

  ws.columns = columns;

  ws.addRows(data);

  // create folder if does not exist
  if(!fs.existsSync(path))
    fs.mkdirSync(join(path))

  await wb.csv.writeFile(
    join(
      // process.cwd(),
      `${path}/${_.join(_.split(_.toLower(sheetLabel), " "), "-")}.csv`
    )
  );
}

export function findAgeRange(age) {
  if (_.isEmpty(age) || !moment(age, "YYYY-MM-DD").isValid()) return "";

  age = moment().diff(age, "years");

  if (age < 0) throw new Error("Invalid date of birth");

  const range = _.find(ageRanges, (a) => a.min <= age && age <= a.max);

  return `${range.min}-${range.max}`;
}