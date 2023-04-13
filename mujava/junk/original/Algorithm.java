package mujava.junk.original;


public class Algorithm
{
    public void playerLogout(String playerId) {
        if (chatClient != null) {
            players.remove(playerId);
        }
    }
}
