## Requirements

1. node v14 or greater
2. npm v6 or greater

### Installation

```sh
npm install
```

### Network data adjustments

1. User size and network size

```js
// file: src/index.js
// function call: main(userSize, transactionSize);
main(300, 1000);
```

2. Transaction range

```js
// file: src/index.js
// function: generateTransactions
const transaction = {
  service: serviceAccorderie[(0, randomizer(0, serviceAccorderie.length - 1))],
  vendeur: `${vendeur.id} ${vendeur.prenom}`,
  acheteur: `${acheteur.id} ${acheteur.prenom}`,
  //FIXME: adjust transaction range END and/or START date below this line i.e: END: moment("2022-06-01"), START: moment("2022-04-01")
  date: randomDate(moment("2022-03-01"), moment("2022-01-01")).format(
    "YYYY-MM-DD"
  ),
  duree: timeRandomizer(),
};
```

3. User age range

```js
// file: src/index.js
// function: generateUsers
const user = {
  nom: faker.name.firstName(),
  prenom: faker.name.lastName(),
  //TODO: Adjust user age range below this line. The example below generates users between 35 - 92 years old
  age: dateRandomizer(moment("1987-01-01"), moment("1930-01-01")),
  address: fakePostCode(),
  genre: faker.name.gender(true),
};
```

### Execution

```sh
npm start
```

### Output

1. Users csv file under dist/ folder containing users' data such as `name`, `last name`, `age`, `gender` and `address` (postal code)
2. Transactions cvs file under dist/ folder containing transactions between users with information such as `vendor`, `buyer`, `service type`, `duration` and transaction `date`
