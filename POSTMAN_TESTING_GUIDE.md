# Garmin AI Chat API - Postman Testing Guide

This guide helps you test the Garmin AI Chat API using the provided Postman collection.

## 🚀 Quick Setup

### 1. Import Collection and Environment

1. **Import Collection**: Import `Garmin_AI_Chat_API.postman_collection.json`
2. **Import Environment**: Import `Garmin_AI_Chat_Local.postman_environment.json`
3. **Select Environment**: Choose "Garmin AI Chat - Local" from the environment dropdown

### 2. Configure Variables

Before testing, update these environment variables:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `base_url` | API server URL | `http://localhost:8000` |
| `user_email` | Test user email | `test@example.com` |
| `user_password` | Test user password | `testpassword123` |
| `garmin_username` | Your Garmin Connect username | `your_garmin_email@example.com` |
| `garmin_password` | Your Garmin Connect password | `your_garmin_password` |

> ⚠️ **Important**: Use real Garmin Connect credentials for testing activity sync functionality.

## 🧪 Testing Workflow

### Phase 1: Server Health Check

**Run these requests first:**
1. `Health & Info` → `Root Endpoint`
2. `Health & Info` → `API Info`  
3. `Health & Info` → `Health Check`

**Expected Results:**
- All requests return 200 status
- Health check shows `"database": "healthy"`
- API info displays correct version and environment

### Phase 2: User Authentication

**Run in order:**
1. `Authentication` → `User Registration`
2. `Authentication` → `User Login` 
3. `Authentication` → `Get Current User`

**What Happens:**
- Creates a test user account
- Authenticates and stores JWT tokens automatically
- Verifies token works for protected endpoints

### Phase 3: Garmin Integration Setup

**Run these requests:**
1. `Authentication` → `Set Garmin Credentials`
2. `Authentication` → `Test Garmin Connection`

**What Happens:**
- Stores encrypted Garmin credentials
- Tests connection to Garmin Connect
- Returns success/failure status

### Phase 4: Activity Synchronization

**Test sync functionality:**
1. `Synchronization` → `Sync Activities - Last 10 Days`
2. `Synchronization` → `Get Sync Status` (monitor progress)
3. `Synchronization` → `Get Sync History`
4. `Synchronization` → `Get Sync Stats`

**What Happens:**
- Starts background sync process
- Returns sync_id for monitoring
- Tracks sync progress and results

### Phase 5: Activity Management

**Test activity endpoints:**
1. `Activities` → `Get Activity Types`
2. `Activities` → `List Activities - All`
3. `Activities` → `Get Activity Details` (uses stored activity_id)
4. `Activities` → `List Activities - Running Only`

**What Happens:**
- Lists available activity types
- Retrieves paginated activity lists
- Shows detailed activity metrics
- Demonstrates filtering capabilities

## 📊 Sample Activity Data

After successful sync, you'll see activities with metrics like:

### Cycling Activities
```json
{
  "activity_type": "virtual_ride",
  "distance": 42240.0,
  "duration": 4267.0,
  "average_speed": 35.57,
  "average_power": 207.0,
  "average_cadence": 84.0,
  "calories": 846
}
```

### Running Activities
```json
{
  "activity_type": "running",
  "distance": 10020.0,
  "duration": 3255.0,
  "average_speed": 11.07,
  "average_heart_rate": 147,
  "elevation_gain": 150.0,
  "calories": 778
}
```

### Swimming Activities
```json
{
  "activity_type": "lap_swimming",
  "distance": 2300.0,
  "duration": 1992.0,
  "average_speed": 4.15,
  "strokes": 1250,
  "pool_length": 25.0,
  "calories": 482
}
```

## 🔍 Test Validation

### Automatic Test Scripts

The collection includes automatic tests that verify:

✅ **Response Status Codes**: Ensures proper HTTP status codes
✅ **Response Structure**: Validates required fields in responses  
✅ **Token Management**: Automatically stores and uses JWT tokens
✅ **Data Persistence**: Stores IDs for dependent requests
✅ **Error Scenarios**: Tests unauthorized access and invalid inputs

### Manual Verification Points

**After User Registration:**
- Check database: `SELECT * FROM users WHERE email = 'test@example.com';`
- Verify password is hashed (not plaintext)

**After Garmin Credentials:**
- Verify credentials are encrypted in database
- Test Garmin connection returns success

**After Activity Sync:**
- Check sync_history table for record
- Verify activities table has new records
- Confirm all activity metrics are stored

## ⚠️ Error Testing

The collection includes error scenario tests:

1. `Error Scenarios` → `Unauthorized Request` (401 expected)
2. `Error Scenarios` → `Invalid Activity ID` (404 expected)  
3. `Error Scenarios` → `Invalid Sync ID` (404 expected)

## 🔧 Troubleshooting

### Common Issues

**"Database unhealthy" in health check:**
- Ensure MySQL is running: `mysql -u root -p`
- Check database exists: `SHOW DATABASES;`
- Verify connection string in `.env`

**"Garmin connection failed":**
- Verify Garmin credentials are correct
- Check for MFA requirements (may need app-specific password)
- Ensure Garmin Connect account is active

**"Sync returns no activities":**
- Check date range (may be outside activity period)
- Verify Garmin account has activities in specified timeframe
- Check sync_history table for error messages

**JWT token expired:**
- Run `Authentication` → `Refresh Token`
- Or re-login with `Authentication` → `User Login`

## 📈 Performance Testing

For load testing:
1. Use Collection Runner with multiple iterations
2. Test with different date ranges
3. Monitor database performance during large syncs
4. Verify memory usage during concurrent requests

## 🎯 Expected Test Results

**Complete test run should show:**
- ✅ 20+ successful requests (200/201/202 status codes)
- ✅ Automatic token management working
- ✅ Database populated with user and activities
- ✅ All health checks passing
- ✅ Error scenarios handled correctly

## 📝 Notes

- **Test Data**: Uses real Garmin data, ensure account has sufficient activities
- **Rate Limiting**: Garmin Connect has rate limits; space out sync requests
- **Data Privacy**: Test credentials are stored in Postman environment
- **Cleanup**: Delete test user from database after testing if needed

Happy testing! 🚀