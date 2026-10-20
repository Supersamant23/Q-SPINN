import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BinanceLiveWS')

async def connect_and_validate():
    # 3-Asset Crypto Basket
    symbols = ['btcusdt', 'ethusdt', 'solusdt']
    
    # We want 100ms order book depth (top 20 levels) and live trades
    streams = []
    for s in symbols:
        streams.append(f"{s}@depth20@100ms")
        streams.append(f"{s}@trade")
        
    stream_url = f"wss://stream.binance.com/stream?streams={'/'.join(streams)}"
    logger.info(f"Connecting to Binance WebSocket: {stream_url}")
    
    try:
        async with websockets.connect(stream_url, open_timeout=30) as websocket:
            logger.info("Successfully connected to Binance Live WebSocket!")
            logger.info("Listening for messages to validate the stream...\n")
            
            message_count = 0
            max_messages = 5 # Receive 5 messages then disconnect for validation
            
            while message_count < max_messages:
                response = await websocket.recv()
                data = json.loads(response)
                
                stream = data.get('stream', 'unknown')
                payload = data.get('data', {})
                
                logger.info(f"[Message {message_count + 1}/{max_messages}] Stream: {stream}")
                
                if 'trade' in stream:
                    logger.info(f"   Trade -> Symbol: {payload.get('s')}, Price: {payload.get('p')}, Qty: {payload.get('q')}, Time: {payload.get('T')}")
                elif 'depth' in stream:
                    bids = payload.get('bids', [])
                    asks = payload.get('asks', [])
                    best_bid = bids[0] if bids else None
                    best_ask = asks[0] if asks else None
                    logger.info(f"   Depth -> Best Bid: {best_bid}, Best Ask: {best_ask}, Update ID: {payload.get('lastUpdateId')}")
                
                message_count += 1
                
            logger.info("\nValidation successful. Closing connection gracefully.")
            
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(connect_and_validate())
