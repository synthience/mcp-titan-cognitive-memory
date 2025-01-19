/** @type {import('ts-jest').JestConfigWithTsJest} */
export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true
      },
    ],
  },
  transformIgnorePatterns: [
    'node_modules/(?!(@tensorflow/tfjs)/)'
  ],
  testPathIgnorePatterns: [
    '/node_modules/',
    '/build/'
  ]
}
