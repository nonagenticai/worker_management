#!/usr/bin/env node

/**
 * Fix script to update all test files to properly handle 204 No Content responses
 */

const fs = require('fs');
const path = require('path');

// Get all test files
const testFiles = fs.readdirSync(__dirname)
  .filter(file => file.startsWith('test_') && file.endsWith('.js'));

console.log(`Found ${testFiles.length} test files to update...`);

const fixCode = `
            // Handle 204 No Content responses
            if (response.status === 204) {
                return { success: true };
            }

            const data = await response.json();`;

const searchPattern = /const data = await response\.json\(\);/g;

let updatedCount = 0;

testFiles.forEach(file => {
  const filePath = path.join(__dirname, file);
  let content = fs.readFileSync(filePath, 'utf8');

  // Check if file needs updating
  if (content.includes('await response.json()') && !content.includes('response.status === 204')) {
    // Replace the pattern
    content = content.replace(searchPattern, fixCode);

    // Write back to file
    fs.writeFileSync(filePath, content);
    console.log(`✅ Updated: ${file}`);
    updatedCount++;
  } else if (content.includes('response.status === 204')) {
    console.log(`⏭️  Already fixed: ${file}`);
  } else {
    console.log(`⏭️  Skipped: ${file} (no JSON parsing found)`);
  }
});

console.log(`\n✅ Updated ${updatedCount} files with 204 No Content handling`);
