from llama_stack_client.lib.agents.client_tool import client_tool
import geocoder

@client_tool
def get_location(query: str = "location"):
    """
    Provide the location upon request.

    :param query: The query from user
    :returns: Information about user location
    """
    try:
        g = geocoder.ip('me')
        if g.ok:
            return f"Your current location is: {g.city}, {g.state}, {g.country}" # can be modified to return latitude and longitude if needed
        else:
            return "Unable to determine your location"
    except Exception as e:
        return f"Error getting location: {str(e)}"